"""
Main pipeline orchestrator for ASR data augmentation.
Coordinates data cleaning, splitting, augmentation, and TSV generation.
"""

import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import argparse

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .data_cleaner import DataCleaner
from .data_splitter import DataSplitter
from .audio_downloader import AudioDownloader
from .audio_augmenter import AudioAugmenter


class ASRDataPipeline:
    """Main pipeline for ASR dataset preparation and augmentation."""

    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.stats = {}

        # Create output directory
        self.output_dir = Path(self.config['output']['base_dir']) / self.config['output']['dataset_name']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("ASR Data Augmentation Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Project: {self.config['project']['name']}")
        self.logger.info(f"Author: {self.config['project']['author']}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 60)

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config['logging']
        log_level = getattr(logging, log_config['level'])

        # Create log directory
        log_file = Path(log_config['log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        if not self.config['mlflow']['enabled']:
            self.logger.info("MLflow tracking disabled")
            return

        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not installed, skipping experiment tracking")
            return

        experiment_name = self.config['mlflow']['experiment_name']
        tracking_uri = self.config['mlflow'].get('tracking_uri')

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        # Start MLflow run
        run_name = f"{self.config['output']['dataset_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        # Log configuration
        mlflow.log_params({
            'project_name': self.config['project']['name'],
            'train_ratio': self.config['splitting']['train_ratio'],
            'val_ratio': self.config['splitting']['val_ratio'],
            'test_ratio': self.config['splitting']['test_ratio'],
            'augmentation_enabled': self.config['augmentation']['enabled'],
            'augmentation_factor': self.config['augmentation']['factor'],
            'augmentation_strategy': self.config['augmentation']['strategy'],
        })

        self.logger.info(f"MLflow run started: {run_name}")

    def run(self) -> Dict:
        """
        Execute the complete pipeline.

        Returns:
            Dictionary with pipeline statistics
        """
        start_time = datetime.now()
        self.logger.info(f"Pipeline started at: {start_time}")

        # Setup MLflow
        self.setup_mlflow()

        # Step 1: Data Cleaning
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 1: Data Cleaning")
        self.logger.info("=" * 60)

        cleaner = DataCleaner(self.config)
        df_clean, cleaning_stats = cleaner.clean(self.config['input']['csv_path'])
        self.stats['cleaning'] = cleaning_stats

        # Log to MLflow
        if self.config['mlflow']['enabled'] and MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'initial_samples': cleaning_stats['cleaning']['initial_count'],
                'cleaned_samples': cleaning_stats['cleaning']['final_count'],
                'duplicates_removed': cleaning_stats['cleaning']['removed_count'],
            })

        # Step 2: Audio Download Preparation
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 2: Audio Download Preparation")
        self.logger.info("=" * 60)

        downloader = AudioDownloader(self.config)
        df_remapped, audio_stats = downloader.prepare_audio(df_clean, auto_download=False)
        self.stats['audio'] = audio_stats

        # Step 3: Dataset Splitting
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 3: Dataset Splitting")
        self.logger.info("=" * 60)

        splitter = DataSplitter(self.config)
        train_df, val_df, test_df, split_stats = splitter.split_dataset(df_remapped)
        self.stats['splitting'] = split_stats

        # Log to MLflow
        if self.config['mlflow']['enabled'] and MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'train_samples_before_aug': split_stats['train_samples'],
                'val_samples': split_stats['val_samples'],
                'test_samples': split_stats['test_samples'],
            })

        # Step 4: Audio Augmentation (Training set only)
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 4: Audio Augmentation")
        self.logger.info("=" * 60)

        augmenter = AudioAugmenter(self.config)

        # Check which splits to augment
        apply_to = self.config['augmentation']['apply_to']

        if apply_to == 'train' or apply_to == 'all':
            train_df_aug, aug_stats = augmenter.augment_dataset(train_df, self.output_dir)
            self.stats['augmentation'] = aug_stats

            # Log to MLflow
            if self.config['mlflow']['enabled'] and MLFLOW_AVAILABLE and aug_stats.get('augmentation_enabled'):
                mlflow.log_metrics({
                    'train_samples_after_aug': aug_stats['total_samples'],
                    'augmented_samples': aug_stats['augmented_samples'],
                    'actual_augmentation_factor': aug_stats['actual_factor'],
                })
        else:
            train_df_aug = train_df
            self.logger.info("Augmentation not applied to training set")

        # Step 5: Save TSV Files
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 5: Saving TSV Files")
        self.logger.info("=" * 60)

        splitter.save_tsv(train_df_aug, self.output_dir / "train.tsv", "train")
        splitter.save_tsv(val_df, self.output_dir / "val.tsv", "validation")
        splitter.save_tsv(test_df, self.output_dir / "test.tsv", "test")

        # Step 6: Save Statistics
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 6: Saving Statistics")
        self.logger.info("=" * 60)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.stats['pipeline'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'config': self.config
        }

        # Save stats to JSON
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)

        self.logger.info(f"Statistics saved to: {stats_file}")

        # Log to MLflow
        if self.config['mlflow']['enabled'] and MLFLOW_AVAILABLE:
            # mlflow.log_artifact(str(stats_file))  # Disabled due to permission issues
            mlflow.log_metric('pipeline_duration_seconds', duration)
            mlflow.end_run()

        # Print summary
        self.print_summary()

        return self.stats

    def print_summary(self) -> None:
        """Print pipeline execution summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)

        cleaning = self.stats['cleaning']
        splitting = self.stats['splitting']

        self.logger.info(f"Input samples: {cleaning['cleaning']['initial_count']}")
        self.logger.info(f"After cleaning: {cleaning['cleaning']['final_count']}")
        self.logger.info(f"  - Duplicates removed: {cleaning['cleaning']['removed_count']}")

        self.logger.info(f"\nDataset splits:")
        self.logger.info(f"  - Train: {splitting['train_samples']} samples")
        self.logger.info(f"  - Validation: {splitting['val_samples']} samples")
        self.logger.info(f"  - Test: {splitting['test_samples']} samples")

        if 'augmentation' in self.stats and self.stats['augmentation'].get('augmentation_enabled'):
            aug = self.stats['augmentation']
            self.logger.info(f"\nAugmentation:")
            self.logger.info(f"  - Original train samples: {aug['original_samples']}")
            self.logger.info(f"  - Augmented samples: {aug['augmented_samples']}")
            self.logger.info(f"  - Total train samples: {aug['total_samples']}")
            self.logger.info(f"  - Augmentation factor: {aug['actual_factor']:.2f}x")

        self.logger.info(f"\nOutput directory: {self.output_dir}")
        self.logger.info(f"Files generated:")
        self.logger.info(f"  - {self.output_dir / 'train.tsv'}")
        self.logger.info(f"  - {self.output_dir / 'val.tsv'}")
        self.logger.info(f"  - {self.output_dir / 'test.tsv'}")
        self.logger.info(f"  - {self.output_dir / 'stats.json'}")

        pipeline = self.stats['pipeline']
        self.logger.info(f"\nPipeline duration: {pipeline['duration_seconds']:.2f} seconds")

        self.logger.info("=" * 60)
        self.logger.info("Pipeline completed successfully!")
        self.logger.info("=" * 60)

        # Audio download reminder
        if self.stats['audio']['missing_files'] > 0:
            self.logger.warning("\n" + "!" * 60)
            self.logger.warning("IMPORTANT: Audio files need to be downloaded!")
            self.logger.warning(f"Run: bash {self.output_dir.parent / 'download_audio.sh'}")
            self.logger.warning("!" * 60)


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description='ASR Data Augmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python pipeline.py

  # Run with custom config
  python pipeline.py --config my_config.yaml

  # Show config and exit
  python pipeline.py --show-config
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show configuration and exit'
    )

    args = parser.parse_args()

    # Load and optionally display config
    if args.show_config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(yaml.dump(config, default_flow_style=False))
        return

    # Run pipeline
    pipeline = ASRDataPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
