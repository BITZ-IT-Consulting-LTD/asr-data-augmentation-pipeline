"""
Audio augmentation module for ASR dataset pipeline.
Implements configurable audio augmentation techniques.
"""

import pandas as pd
import numpy as np
import logging
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import random
import nlpaug.augmenter.audio as naa


class AudioAugmenter:
    """Apply audio augmentation techniques to ASR dataset."""

    def __init__(self, config: Dict):
        """
        Initialize AudioAugmenter with configuration.

        Args:
            config: Configuration dictionary containing augmentation parameters
        """
        self.config = config
        self.aug_config = config['augmentation']
        self.logger = logging.getLogger(__name__)

        # Set random seed for reproducibility
        self.random_seed = config['splitting']['random_seed']
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Initialize nlpaug augmenters
        self.nlpaug_augmenters = {}
        self._initialize_nlpaug_augmenters()

        # Load probability weights
        self.probabilities = self.aug_config.get('probabilities', {})

    def _initialize_nlpaug_augmenters(self):
        """
        Initialize nlpaug augmenters based on configuration.

        Creates augmenters for:
        - colored_noise: NoiseAug for white/pink/brown noise
        - crop_audio: CropAug for VoIP packet loss simulation
        - vtlp: VtlpAug for speaker variability
        - mask_audio: MaskAug for codec artifacts
        """
        techniques_config = self.aug_config.get('techniques', {})

        # Colored Noise (replaces background_noise)
        if techniques_config.get('colored_noise', {}).get('enabled'):
            config = techniques_config['colored_noise']
            try:
                # nlpaug NoiseAug doesn't need sampling_rate parameter
                self.nlpaug_augmenters['colored_noise'] = naa.NoiseAug(
                    zone=tuple(config.get('zone', [0.0, 1.0])),
                    coverage=config.get('coverage', 1.0),
                    color='random'  # Randomly picks from white/pink/brown
                )
                self.logger.info("Initialized colored_noise augmenter (nlpaug)")
            except Exception as e:
                self.logger.error(f"Failed to initialize colored_noise augmenter: {e}")

        # Crop Audio (Packet Loss)
        if techniques_config.get('crop_audio', {}).get('enabled'):
            config = techniques_config['crop_audio']
            try:
                self.nlpaug_augmenters['crop_audio'] = naa.CropAug(
                    sampling_rate=16000,
                    zone=tuple(config.get('zone', [0.1, 0.9])),
                    coverage=config.get('coverage', 0.15)
                )
                self.logger.info("Initialized crop_audio augmenter (nlpaug)")
            except Exception as e:
                self.logger.error(f"Failed to initialize crop_audio augmenter: {e}")

        # VTLP (Speaker Variability)
        if techniques_config.get('vtlp', {}).get('enabled'):
            config = techniques_config['vtlp']
            try:
                self.nlpaug_augmenters['vtlp'] = naa.VtlpAug(
                    sampling_rate=16000,
                    zone=tuple(config.get('zone', [0.2, 0.8])),
                    coverage=config.get('coverage', 0.1),
                    factor=tuple(config.get('factor_range', [0.9, 1.1]))
                )
                self.logger.info("Initialized vtlp augmenter (nlpaug)")
            except Exception as e:
                self.logger.error(f"Failed to initialize vtlp augmenter: {e}")

        # Mask Audio (Codec Artifacts)
        if techniques_config.get('mask_audio', {}).get('enabled'):
            config = techniques_config['mask_audio']
            try:
                self.nlpaug_augmenters['mask_audio'] = naa.MaskAug(
                    sampling_rate=16000,
                    zone=tuple(config.get('zone', [0.1, 0.9])),
                    coverage=config.get('coverage', 0.1),
                    mask_with_noise=config.get('mask_with_noise', True)
                )
                self.logger.info("Initialized mask_audio augmenter (nlpaug)")
            except Exception as e:
                self.logger.error(f"Failed to initialize mask_audio augmenter: {e}")

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr

    def save_audio(self, audio: np.ndarray, sr: int, output_path: str) -> None:
        """
        Save audio to file.

        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Path to save audio
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr)

    def pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply pitch shifting to audio.

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Pitch-shifted audio
        """
        config = self.aug_config['techniques']['pitch_shift']
        if not config['enabled']:
            return audio

        n_steps_range = config['n_steps_range']
        n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])

        augmented = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return augmented

    def time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply time stretching to audio.

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Time-stretched audio
        """
        config = self.aug_config['techniques']['time_stretch']
        if not config['enabled']:
            return audio

        rate_range = config['rate_range']
        rate = np.random.uniform(rate_range[0], rate_range[1])

        augmented = librosa.effects.time_stretch(audio, rate=rate)
        return augmented

    def volume_variation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply volume variation to audio.

        Args:
            audio: Input audio
            sr: Sample rate (unused but kept for consistency)

        Returns:
            Volume-adjusted audio
        """
        config = self.aug_config['techniques']['volume_variation']
        if not config['enabled']:
            return audio

        gain_db_range = config['gain_db_range']
        gain_db = np.random.uniform(gain_db_range[0], gain_db_range[1])

        # Convert dB to linear gain
        gain_linear = 10 ** (gain_db / 20)

        augmented = audio * gain_linear
        # Clip to prevent overflow
        augmented = np.clip(augmented, -1.0, 1.0)

        return augmented

    def _is_technique_enabled(self, technique: str) -> bool:
        """
        Check if a technique is enabled in configuration.

        Args:
            technique: Technique name

        Returns:
            True if technique is enabled, False otherwise
        """
        # Check if it's an nlpaug augmenter (already initialized = enabled)
        if technique in self.nlpaug_augmenters:
            return True

        # Check librosa-based techniques
        technique_config = self.aug_config.get('techniques', {}).get(technique, {})
        return technique_config.get('enabled', False)

    def _get_all_enabled_techniques(self) -> List[str]:
        """
        Get list of all enabled techniques.

        Returns:
            List of enabled technique names
        """
        enabled = list(self.nlpaug_augmenters.keys())

        # Add librosa-based techniques if enabled
        for tech in ['pitch_shift', 'time_stretch', 'volume_variation']:
            if self.aug_config.get('techniques', {}).get(tech, {}).get('enabled'):
                enabled.append(tech)

        return enabled

    def select_augmentation_techniques_weighted(self) -> List[str]:
        """
        Select ONE augmentation technique based on probability weights.

        Strategy: One technique per augmentation for clarity and better training.
        Each augmented version tests one specific robustness (noise, pitch, speed, etc.)

        Returns:
            List with single technique name (kept as list for API compatibility)
        """
        # Build a weighted list of enabled techniques
        enabled_techniques = []
        weights = []

        for technique, probability in self.probabilities.items():
            # Check if technique is enabled
            if not self._is_technique_enabled(technique):
                continue

            enabled_techniques.append(technique)
            weights.append(probability)

        # Select ONE technique based on weighted probabilities
        if enabled_techniques:
            # Use numpy's weighted random choice
            selected_technique = np.random.choice(
                enabled_techniques,
                p=np.array(weights) / np.sum(weights)  # Normalize to sum to 1.0
            )
            return [selected_technique]
        else:
            # Fallback: if no techniques enabled, return empty
            self.logger.warning("No augmentation techniques enabled!")
            return []

    def add_background_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        DEPRECATED: Use colored_noise (nlpaug) instead.

        Add background noise to audio (legacy librosa implementation).

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Audio with added noise
        """
        config = self.aug_config['techniques'].get('background_noise', {})
        if not config.get('enabled'):
            return audio

        # Generate white noise
        noise = np.random.normal(0, 1, len(audio))

        # Get SNR (Signal-to-Noise Ratio) in dB
        snr_db_range = config.get('snr_db_range', [15, 25])
        snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])

        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Scale noise to achieve desired SNR
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))

        augmented = audio + noise
        # Clip to prevent overflow
        augmented = np.clip(augmented, -1.0, 1.0)

        return augmented

    def apply_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        techniques: List[str]
    ) -> np.ndarray:
        """
        Apply a single augmentation technique to audio.
        Handles both nlpaug augmenters and librosa functions.

        Strategy: One technique per augmentation for maximum clarity.
        Each augmented version has one specific characteristic (noise, pitch change, etc.)

        Args:
            audio: Input audio
            sr: Sample rate
            techniques: List of technique names (should contain exactly 1 technique)

        Returns:
            Augmented audio
        """
        # Expect exactly one technique
        if len(techniques) == 0:
            self.logger.warning("No technique provided, returning original audio")
            return audio.copy()

        if len(techniques) > 1:
            self.logger.warning(
                f"Multiple techniques provided: {techniques}. "
                f"Using only the first one: {techniques[0]}"
            )

        technique = techniques[0]
        augmented = audio.copy()

        try:
            if technique in self.nlpaug_augmenters:
                # Apply nlpaug augmenter
                self.logger.debug(f"Applying nlpaug technique: {technique}")
                augmented = self.nlpaug_augmenters[technique].augment(augmented)
                # nlpaug returns list, convert to numpy array and squeeze extra dimensions
                if isinstance(augmented, list):
                    augmented = np.array(augmented)
                if augmented.ndim > 1:
                    augmented = np.squeeze(augmented)
            else:
                # Apply librosa-based augmentation
                if technique == 'pitch_shift':
                    augmented = self.pitch_shift(augmented, sr)
                elif technique == 'time_stretch':
                    augmented = self.time_stretch(augmented, sr)
                elif technique == 'volume_variation':
                    augmented = self.volume_variation(augmented, sr)
                elif technique == 'background_noise':
                    # Legacy support - warn user to migrate to colored_noise
                    self.logger.warning(
                        "background_noise is deprecated. Please use colored_noise instead."
                    )
                    augmented = self.add_background_noise(augmented, sr)
                else:
                    self.logger.warning(f"Unknown augmentation technique: {technique}")
                    return audio.copy()

        except Exception as e:
            self.logger.error(f"Failed to apply {technique}: {e}")
            # Return original audio if augmentation fails
            return audio.copy()

        return augmented

    def select_augmentation_techniques(self) -> List[str]:
        """
        Select augmentation techniques based on configuration.

        Supports two modes:
        1. Weighted probability selection (if 'probabilities' config exists)
        2. Legacy strategy-based selection (for backward compatibility)

        Returns:
            List of technique names to apply
        """
        # Use weighted selection if probabilities are configured
        if self.probabilities:
            return self.select_augmentation_techniques_weighted()

        # Fallback to legacy strategy-based selection
        self.logger.warning(
            "Using legacy strategy-based selection. "
            "Consider migrating to probability-based selection for better control."
        )

        # Get all enabled techniques (legacy method)
        enabled_techniques = []
        for technique_name, technique_config in self.aug_config['techniques'].items():
            if technique_config.get('enabled', False):
                enabled_techniques.append(technique_name)

        if not enabled_techniques:
            return []

        strategy = self.aug_config.get('strategy', 'random')

        if strategy == 'all':
            return enabled_techniques
        elif strategy == 'single':
            return [random.choice(enabled_techniques)]
        elif strategy == 'random':
            min_techniques = self.aug_config.get('min_techniques_per_sample', 1)
            max_techniques = self.aug_config.get('max_techniques_per_sample', 2)

            # Ensure we don't exceed available techniques
            max_techniques = min(max_techniques, len(enabled_techniques))
            min_techniques = min(min_techniques, len(enabled_techniques))

            num_techniques = random.randint(min_techniques, max_techniques)
            return random.sample(enabled_techniques, num_techniques)
        else:
            self.logger.warning(f"Unknown strategy: {strategy}, using 'random'")
            return random.sample(enabled_techniques, 1)

    def augment_sample(
        self,
        audio_path: str,
        output_path: str,
        aug_id: int
    ) -> Tuple[str, float, List[str]]:
        """
        Create one augmented version of an audio file.

        Args:
            audio_path: Path to input audio
            output_path: Path to save augmented audio
            aug_id: Augmentation ID for this sample

        Returns:
            Tuple of (output_path, duration, techniques_applied)
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)

        # Select techniques
        techniques = self.select_augmentation_techniques()

        # Apply augmentation
        augmented = self.apply_augmentation(audio, sr, techniques)

        # Save augmented audio
        self.save_audio(augmented, sr, output_path)

        # Calculate duration
        duration = len(augmented) / sr

        return output_path, duration, techniques

    def augment_dataset(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Augment entire dataset according to configuration.

        Args:
            df: Input DataFrame
            output_dir: Directory for augmented audio files

        Returns:
            Tuple of (augmented DataFrame with all samples, statistics)
        """
        if not self.aug_config['enabled']:
            self.logger.info("Augmentation disabled, skipping")
            return df, {'augmentation_enabled': False}

        self.logger.info("=" * 60)
        self.logger.info("Starting audio augmentation")
        self.logger.info("=" * 60)

        augmentation_factor = self.aug_config['factor']
        original_count = len(df)

        # Calculate number of augmented samples needed
        # factor=1 means no augmentation, factor=2 means double, factor=3 means triple
        num_augmented_per_sample = augmentation_factor - 1

        if num_augmented_per_sample <= 0:
            self.logger.info(f"Augmentation factor is {augmentation_factor}, no augmentation needed")
            return df, {'augmentation_enabled': True, 'augmentation_factor': augmentation_factor}

        self.logger.info(f"Augmentation factor: {augmentation_factor}x")
        self.logger.info(f"Original samples: {original_count}")
        self.logger.info(f"Augmented samples per original: {num_augmented_per_sample}")
        self.logger.info(f"Expected total: {original_count * augmentation_factor}")

        # Create augmented samples
        augmented_rows = []
        technique_counter = {}

        aug_audio_dir = output_dir / "audio" / "augmented"
        aug_audio_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting audio"):
            original_path = row['audio']
            transcription = row['transcription']
            original_filename = Path(original_path).stem

            # Create augmented versions
            for aug_id in range(num_augmented_per_sample):
                # Generate augmented filename
                aug_filename = f"{original_filename}_aug{aug_id+1}.wav"
                aug_path = aug_audio_dir / aug_filename

                try:
                    # Create augmented audio
                    output_path, duration, techniques = self.augment_sample(
                        original_path,
                        str(aug_path),
                        aug_id
                    )

                    # Track technique usage
                    for tech in techniques:
                        technique_counter[tech] = technique_counter.get(tech, 0) + 1

                    # Add to augmented rows
                    augmented_rows.append({
                        'audio': output_path,
                        'transcription': transcription,
                        'duration': duration
                    })

                except Exception as e:
                    self.logger.error(f"Failed to augment {original_path}: {e}")

        # Combine original and augmented data
        df_augmented = pd.DataFrame(augmented_rows)
        df_combined = pd.concat([df, df_augmented], ignore_index=True)

        # Shuffle the combined dataset for better mini-batch diversity during training
        # This prevents augmented versions of the same audio from appearing in the same batch
        shuffle_enabled = self.aug_config.get('shuffle_after_augmentation', True)
        if shuffle_enabled:
            self.logger.info("Shuffling augmented dataset for better mini-batch diversity...")
            df_combined = df_combined.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            self.logger.info("Dataset shuffled successfully")

        # Calculate statistics
        stats = {
            'augmentation_enabled': True,
            'augmentation_factor': augmentation_factor,
            'original_samples': original_count,
            'augmented_samples': len(df_augmented),
            'total_samples': len(df_combined),
            'actual_factor': len(df_combined) / original_count,
            'techniques_applied': technique_counter,
            'strategy': self.aug_config['strategy'],
            'shuffled': shuffle_enabled
        }

        self.logger.info("=" * 60)
        self.logger.info("Augmentation complete")
        self.logger.info(f"Original samples: {stats['original_samples']}")
        self.logger.info(f"Augmented samples: {stats['augmented_samples']}")
        self.logger.info(f"Total samples: {stats['total_samples']}")
        self.logger.info(f"Actual factor: {stats['actual_factor']:.2f}x")
        self.logger.info(f"Technique usage: {technique_counter}")
        self.logger.info(f"Shuffled: {shuffle_enabled}")
        self.logger.info("=" * 60)

        return df_combined, stats


if __name__ == "__main__":
    # Example usage
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # Create sample data
    data = {
        'audio': ['sample.wav'],
        'transcription': ['Test transcription'],
        'duration': [5.0]
    }
    df = pd.DataFrame(data)

    augmenter = AudioAugmenter(config)
    print(f"Augmentation strategy: {augmenter.aug_config['strategy']}")
    print(f"Augmentation factor: {augmenter.aug_config['factor']}")
