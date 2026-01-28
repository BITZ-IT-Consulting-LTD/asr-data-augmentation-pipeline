"""
Data cleaning module for ASR dataset pipeline.
Handles deduplication, column filtering, and data validation.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple


class DataCleaner:
    """Clean and preprocess Label Studio ASR exports."""

    def __init__(self, config: Dict):
        """
        Initialize DataCleaner with configuration.

        Args:
            config: Configuration dictionary containing cleaning parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove duplicate audio files.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (cleaned DataFrame, statistics dict)
        """
        initial_count = len(df)
        strategy = self.config['cleaning']['duplicate_strategy']

        # Check for duplicates
        duplicate_count = df['audio'].duplicated().sum()
        self.logger.info(f"Found {duplicate_count} duplicate audio files")

        if self.config['cleaning']['remove_duplicates']:
            if strategy == 'keep_first':
                df_cleaned = df.drop_duplicates(subset=['audio'], keep='first')
            elif strategy == 'keep_last':
                df_cleaned = df.drop_duplicates(subset=['audio'], keep='last')
            else:
                raise ValueError(f"Unknown duplicate strategy: {strategy}")

            removed_count = initial_count - len(df_cleaned)
            self.logger.info(f"Removed {removed_count} duplicate records (strategy: {strategy})")
        else:
            df_cleaned = df.copy()
            removed_count = 0
            self.logger.info("Duplicate removal disabled")

        stats = {
            'initial_count': initial_count,
            'duplicate_count': duplicate_count,
            'removed_count': removed_count,
            'final_count': len(df_cleaned)
        }

        return df_cleaned, stats

    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only specified columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with filtered columns
        """
        keep_columns = self.config['cleaning']['keep_columns']
        self.logger.info(f"Keeping columns: {keep_columns}")

        missing_columns = [col for col in keep_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        df_filtered = df[keep_columns].copy()
        dropped_columns = set(df.columns) - set(keep_columns)
        self.logger.info(f"Dropped columns: {dropped_columns}")

        return df_filtered

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality and return statistics.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duration_stats': df['duration'].describe().to_dict(),
            'transcription_length_stats': df['transcription'].str.len().describe().to_dict(),
            'unique_audio_files': df['audio'].nunique()
        }

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            self.logger.warning(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")

        # Check for empty transcriptions
        empty_transcriptions = (df['transcription'].str.len() == 0).sum()
        if empty_transcriptions > 0:
            self.logger.warning(f"Found {empty_transcriptions} empty transcriptions")
            stats['empty_transcriptions'] = empty_transcriptions

        # Duration validation
        if df['duration'].min() < 1:
            self.logger.warning(f"Found very short audio files (min: {df['duration'].min():.2f}s)")
        if df['duration'].max() > 30:
            self.logger.warning(f"Found very long audio files (max: {df['duration'].max():.2f}s)")

        self.logger.info(f"Validation complete: {stats['total_records']} valid records")

        return stats

    def clean(self, csv_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute full cleaning pipeline.

        Args:
            csv_path: Path to input CSV file

        Returns:
            Tuple of (cleaned DataFrame, statistics dict)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting data cleaning pipeline")
        self.logger.info("=" * 60)

        # Load data
        df = self.load_csv(csv_path)

        # Remove duplicates
        df_clean, dup_stats = self.remove_duplicates(df)

        # Filter columns
        df_clean = self.filter_columns(df_clean)

        # Validate
        validation_stats = self.validate_data(df_clean)

        # Combine statistics
        all_stats = {
            'cleaning': dup_stats,
            'validation': validation_stats
        }

        self.logger.info("=" * 60)
        self.logger.info(f"Data cleaning complete: {len(df_clean)} records ready")
        self.logger.info("=" * 60)

        return df_clean, all_stats


if __name__ == "__main__":
    # Example usage
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    cleaner = DataCleaner(config)
    df_clean, stats = cleaner.clean(config['input']['csv_path'])

    print("\nCleaning Statistics:")
    print(f"Initial records: {stats['cleaning']['initial_count']}")
    print(f"Duplicates found: {stats['cleaning']['duplicate_count']}")
    print(f"Duplicates removed: {stats['cleaning']['removed_count']}")
    print(f"Final records: {stats['cleaning']['final_count']}")
    print(f"\nDuration range: {stats['validation']['duration_stats']['min']:.2f}s - {stats['validation']['duration_stats']['max']:.2f}s")
