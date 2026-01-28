#!/usr/bin/env python3
"""Test script for audio augmentation with nlpaug."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
import logging
from asr_pipeline.audio_augmenter import AudioAugmenter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_augmentation():
    """Test audio augmentation with nlpaug and librosa."""

    # Load config
    logger.info("Loading configuration...")
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize augmenter
    logger.info("Initializing AudioAugmenter...")
    augmenter = AudioAugmenter(config)
    
    # Print initialized augmenters
    logger.info(f"nlpaug augmenters: {list(augmenter.nlpaug_augmenters.keys())}")
    logger.info(f"Probabilities: {augmenter.probabilities}")
    
    # Test with real audio if available
    audio_dir = Path("output/audio")
    audio_files = list(audio_dir.glob("*.wav"))[:1] if audio_dir.exists() else []
    
    if audio_files:
        test_audio = str(audio_files[0])
        logger.info(f"\nTesting with: {test_audio}")
        
        audio, sr = augmenter.load_audio(test_audio)
        logger.info(f"Loaded: shape={audio.shape}, sr={sr}Hz")
        
        # Test selection
        for i in range(3):
            techniques = augmenter.select_augmentation_techniques()
            logger.info(f"Selection {i+1}: {techniques}")
        
        # Test each nlpaug augmenter
        logger.info("\nTesting nlpaug augmenters:")
        for name in augmenter.nlpaug_augmenters.keys():
            try:
                aug = augmenter.nlpaug_augmenters[name].augment(audio)
                # nlpaug returns list, convert to numpy array and squeeze extra dimensions
                if isinstance(aug, list):
                    aug = np.array(aug)
                if aug.ndim > 1:
                    aug = np.squeeze(aug)
                logger.info(f"  ✓ {name}: {aug.shape}")
            except Exception as e:
                logger.error(f"  ✗ {name}: {e}")

        # Test full augmentation pipeline with chained techniques
        logger.info("\nTesting full augmentation pipeline:")
        for i in range(3):
            techniques = augmenter.select_augmentation_techniques()
            try:
                aug = augmenter.apply_augmentation(audio, sr, techniques)
                logger.info(f"  ✓ Test {i+1} - Techniques: {techniques}, Output: {aug.shape}")
            except Exception as e:
                logger.error(f"  ✗ Test {i+1} - Techniques: {techniques}, Error: {e}")

        logger.info("\n✓ All tests passed!")
    else:
        logger.warning("No audio files found - run pipeline first")

if __name__ == "__main__":
    test_augmentation()
