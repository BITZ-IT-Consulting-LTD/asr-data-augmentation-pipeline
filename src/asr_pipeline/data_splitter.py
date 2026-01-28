"""
Data splitting module for ASR dataset pipeline.
Handles train/validation/test splitting with configurable ratios.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Split dataset into train/validation/test sets."""

    def __init__(self, config: Dict):
        """
        Initialize DataSplitter with configuration.

        Args:
            config: Configuration dictionary containing splitting parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def split_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df, statistics_dict)
        """
        split_config = self.config['splitting']

        train_ratio = split_config['train_ratio']
        val_ratio = split_config['val_ratio']
        test_ratio = split_config['test_ratio']
        random_seed = split_config['random_seed']
        shuffle = split_config['shuffle']

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        self.logger.info("=" * 60)
        self.logger.info("Splitting dataset")
        self.logger.info("=" * 60)
        self.logger.info(f"Total samples: {len(df)}")
        self.logger.info(
            f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}"
        )

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=shuffle
        )

        # Second split: separate train and validation from remaining data
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=random_seed,
            shuffle=shuffle
        )

        # Calculate statistics
        stats = {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_percentage': len(train_df) / len(df) * 100,
            'val_percentage': len(val_df) / len(df) * 100,
            'test_percentage': len(test_df) / len(df) * 100,
            'train_duration_total': train_df['duration'].sum(),
            'val_duration_total': val_df['duration'].sum(),
            'test_duration_total': test_df['duration'].sum(),
        }

        self.logger.info(f"Train set: {len(train_df)} samples ({stats['train_percentage']:.1f}%)")
        self.logger.info(f"Validation set: {len(val_df)} samples ({stats['val_percentage']:.1f}%)")
        self.logger.info(f"Test set: {len(test_df)} samples ({stats['test_percentage']:.1f}%)")
        self.logger.info(
            f"Total duration - Train: {stats['train_duration_total']:.1f}s, "
            f"Val: {stats['val_duration_total']:.1f}s, "
            f"Test: {stats['test_duration_total']:.1f}s"
        )
        self.logger.info("=" * 60)

        return train_df, val_df, test_df, stats

    def save_tsv(
        self,
        df: pd.DataFrame,
        output_path: Path,
        split_name: str
    ) -> None:
        """
        Save DataFrame to TSV file.

        Args:
            df: DataFrame to save
            output_path: Path to save TSV file
            split_name: Name of the split (train/val/test)
        """
        tsv_config = self.config['tsv_format']
        separator = tsv_config['separator']
        include_header = tsv_config['include_header']

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to TSV
        df.to_csv(
            output_path,
            sep=separator,
            index=False,
            header=include_header
        )

        self.logger.info(f"Saved {split_name} set to: {output_path} ({len(df)} samples)")

    def process_and_save(
        self, df: pd.DataFrame, output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Split dataset and save all splits to TSV files.

        Args:
            df: Input DataFrame
            output_dir: Base output directory

        Returns:
            Tuple of (train_df, val_df, test_df, statistics_dict)
        """
        # Split the data
        train_df, val_df, test_df, stats = self.split_dataset(df)

        # Save TSV files
        self.save_tsv(train_df, output_dir / "train.tsv", "train")
        self.save_tsv(val_df, output_dir / "val.tsv", "validation")
        self.save_tsv(test_df, output_dir / "test.tsv", "test")

        return train_df, val_df, test_df, stats


if __name__ == "__main__":
    # Example usage
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # Create sample data
    data = {
        'audio': [f'audio_{i}.wav' for i in range(100)],
        'transcription': [f'Sample transcription {i}' for i in range(100)],
        'duration': [5.0] * 100
    }
    df = pd.DataFrame(data)

    # Split
    splitter = DataSplitter(config)
    train_df, val_df, test_df, stats = splitter.split_dataset(df)

    print("\nSplit Statistics:")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
