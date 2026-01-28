"""
Audio downloader module for ASR dataset pipeline.
Downloads audio files from remote EC2 server and remaps paths.
"""

import pandas as pd
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


class AudioDownloader:
    """Download and manage audio files from remote server."""

    def __init__(self, config: Dict):
        """
        Initialize AudioDownloader with configuration.

        Args:
            config: Configuration dictionary containing input/output parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.remote_path = config['input']['audio_remote_path']
        self.ssh_key = Path(config['input']['ssh_key_path']).expanduser()
        self.label_studio_prefix = config['input']['label_studio_prefix']
        self.local_audio_dir = Path(config['output']['base_dir']) / Path(config['input']['local_audio_dir'])

    def generate_download_script(self, df: pd.DataFrame, script_path: Path) -> None:
        """
        Generate a bash script to download only the specific audio files we need using rsync.

        Args:
            df: DataFrame with audio paths
            script_path: Path where to save the download script
        """
        self.logger.info("Generating audio download script")

        # Get unique audio files (just filenames, not full paths)
        audio_files = df['audio'].apply(lambda x: Path(x).name).unique()

        # Create a file list for rsync
        file_list_path = script_path.parent / "audio_files_to_download.txt"
        with open(file_list_path, 'w') as f:
            for filename in audio_files:
                f.write(f"{filename}\n")

        self.logger.info(f"Created file list with {len(audio_files)} files: {file_list_path}")

        script_content = f"""#!/bin/bash
# Audio Download Script for ASR Dataset
# Generated automatically by audio_downloader.py
# This script downloads ONLY the specific audio files needed for the dataset

set -e  # Exit on error

SSH_KEY="{self.ssh_key}"
REMOTE_HOST="ec2-user@ec2-18-177-175-202.ap-northeast-1.compute.amazonaws.com"
REMOTE_DIR="/opt/label-studio/media/project_3"
LOCAL_DIR="{self.local_audio_dir}"
FILE_LIST="{file_list_path}"

echo "========================================"
echo "Audio Download Script"
echo "========================================"
echo "Remote: $REMOTE_HOST:$REMOTE_DIR"
echo "Local: $LOCAL_DIR"
echo "Files to download: {len(audio_files)}"
echo "========================================"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download ONLY the specific audio files we need using rsync
# --files-from: read list of files to transfer from FILE_LIST
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -z: compress during transfer
# -P: show progress and allow resuming
# --relative: use relative path names
# --no-implied-dirs: don't send implied directories

echo "Starting download of {len(audio_files)} specific files..."
echo ""

rsync -avzP \\
    -e "ssh -i $SSH_KEY" \\
    --files-from="$FILE_LIST" \\
    --no-implied-dirs \\
    "$REMOTE_HOST:$REMOTE_DIR/" \\
    "$LOCAL_DIR/"

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"

# Verify downloaded files
EXPECTED_FILES={len(audio_files)}
DOWNLOADED_FILES=$(find "$LOCAL_DIR" -name "*.wav" | wc -l)

echo "Expected files: $EXPECTED_FILES"
echo "Downloaded files: $DOWNLOADED_FILES"

if [ "$DOWNLOADED_FILES" -eq "$EXPECTED_FILES" ]; then
    echo "✓ All files downloaded successfully"
    exit 0
else
    echo "✗ Warning: File count mismatch"
    echo "  Expected: $EXPECTED_FILES"
    echo "  Found: $DOWNLOADED_FILES"
    echo ""
    echo "You can re-run this script to download missing files."
    exit 1
fi
"""

        # Write script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

        self.logger.info(f"Download script saved to: {script_path}")
        self.logger.info(f"To download audio files, run: bash {script_path}")

    def remap_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap audio paths from Label Studio format to local paths.

        Args:
            df: DataFrame with Label Studio audio paths

        Returns:
            DataFrame with remapped local paths
        """
        self.logger.info("Remapping audio paths from Label Studio format to local paths")

        df = df.copy()

        # Extract filename from Label Studio path and create local path
        df['audio'] = df['audio'].apply(
            lambda x: str(self.local_audio_dir / Path(x).name)
        )

        self.logger.info(f"Remapped {len(df)} audio paths")
        self.logger.info(f"Example path: {df['audio'].iloc[0]}")

        return df

    def verify_audio_files(self, df: pd.DataFrame) -> Tuple[List[str], List[str], Dict]:
        """
        Verify that all audio files exist locally.

        Args:
            df: DataFrame with audio paths

        Returns:
            Tuple of (existing_files, missing_files, statistics)
        """
        self.logger.info("Verifying audio files...")

        audio_files = df['audio'].unique()
        existing_files = []
        missing_files = []

        for audio_path in tqdm(audio_files, desc="Verifying files"):
            if Path(audio_path).exists():
                existing_files.append(audio_path)
            else:
                missing_files.append(audio_path)

        stats = {
            'total_files': len(audio_files),
            'existing_files': len(existing_files),
            'missing_files': len(missing_files),
            'completion_percentage': len(existing_files) / len(audio_files) * 100
        }

        self.logger.info(f"Verification complete:")
        self.logger.info(f"  Total files: {stats['total_files']}")
        self.logger.info(f"  Existing: {stats['existing_files']}")
        self.logger.info(f"  Missing: {stats['missing_files']}")
        self.logger.info(f"  Completion: {stats['completion_percentage']:.1f}%")

        if missing_files:
            self.logger.warning(f"Missing {len(missing_files)} audio files. Run the download script first.")
            self.logger.debug(f"First 5 missing files: {missing_files[:5]}")

        return existing_files, missing_files, stats

    def run_download_script(self, script_path: Path) -> bool:
        """
        Execute the download script.

        Args:
            script_path: Path to the bash download script

        Returns:
            True if download was successful, False otherwise
        """
        self.logger.info(f"Running download script: {script_path}")

        try:
            result = subprocess.run(
                ['bash', str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info("Download script output:")
            self.logger.info(result.stdout)

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Download script failed with exit code {e.returncode}")
            self.logger.error(f"Error output: {e.stderr}")
            return False

    def prepare_audio(
        self, df: pd.DataFrame, auto_download: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare audio files: remap paths, generate download script, optionally download.

        Args:
            df: DataFrame with Label Studio audio paths
            auto_download: If True, automatically run download script

        Returns:
            Tuple of (DataFrame with remapped paths, statistics)
        """
        self.logger.info("=" * 60)
        self.logger.info("Preparing audio files")
        self.logger.info("=" * 60)

        # Remap paths
        df_remapped = self.remap_paths(df)

        # Generate download script
        script_path = Path(self.config['output']['base_dir']) / "download_audio.sh"
        self.generate_download_script(df_remapped, script_path)

        # Optionally run download
        if auto_download:
            self.logger.info("Auto-download enabled, running download script...")
            download_success = self.run_download_script(script_path)
            if not download_success:
                self.logger.error("Audio download failed")
        else:
            self.logger.info(f"Audio download script generated: {script_path}")
            self.logger.info("Run the script manually to download audio files")

        # Verify files
        existing, missing, stats = self.verify_audio_files(df_remapped)

        self.logger.info("=" * 60)

        return df_remapped, stats


if __name__ == "__main__":
    # Example usage
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # Create sample data with Label Studio paths
    data = {
        'audio': [
            '/data/media/project_3/audio_1.wav',
            '/data/media/project_3/audio_2.wav',
        ],
        'transcription': ['Sample 1', 'Sample 2'],
        'duration': [5.0, 5.0]
    }
    df = pd.DataFrame(data)

    downloader = AudioDownloader(config)
    df_remapped, stats = downloader.prepare_audio(df, auto_download=False)

    print("\nRemapped paths:")
    print(df_remapped['audio'].head())
