# Audio Augmentation Guide

## Overview

This ASR data augmentation pipeline uses a **hybrid approach** combining nlpaug and librosa to create robust training data for call center ASR systems. The augmentations are designed to simulate real-world call center challenges.

### Architecture

```
AudioAugmenter
├── nlpaug augmenters (call center-specific)
│   ├── colored_noise    - Background noise (white/pink/brown)
│   ├── crop_audio       - VoIP packet loss simulation
│   ├── vtlp             - Speaker variability (vocal tract length)
│   └── mask_audio       - Codec artifacts
└── librosa augmenters (proven techniques)
    ├── pitch_shift      - Pitch variation
    ├── time_stretch     - Speed variation
    └── volume_variation - Volume levels
```

## Why Hybrid nlpaug + librosa?

### nlpaug Advantages
- **Call center-specific**: NoiseAug for realistic background noise
- **VoIP simulation**: CropAug for packet loss scenarios
- **Speaker diversity**: VtlpAug for vocal tract variation
- **Codec artifacts**: MaskAug for compression effects

### librosa Advantages
- **Fine-grained control**: Low-level audio manipulation
- **Proven techniques**: Pitch shift and time stretch
- **Predictable output**: Deterministic transformations

## Configuration

### Probability-Weighted Selection

Each technique is independently applied based on its probability:

```yaml
augmentation:
  probabilities:
    colored_noise: 0.8     # 80% of samples get noise
    crop_audio: 0.4        # 40% get packet loss
    vtlp: 0.5              # 50% get speaker variability
    mask_audio: 0.3        # 30% get codec artifacts
    pitch_shift: 0.3       # 30% get pitch variation
    time_stretch: 0.3      # 30% get speed variation
    volume_variation: 0.7  # 70% get volume changes
```

**Example**: A sample might get:
- colored_noise (80% chance) ✓
- crop_audio (40% chance) ✗
- volume_variation (70% chance) ✓
- Result: 2 techniques applied

### Augmentation Factor

```yaml
augmentation:
  factor: 3  # 3x dataset size (original + 2 augmented versions per sample)
```

- `factor=1`: No augmentation (original only)
- `factor=2`: Double size (original + 1 augmented)
- `factor=3`: Triple size (original + 2 augmented)

## Augmentation Techniques

### 1. Colored Noise (nlpaug)

**Purpose**: Simulate call center background noise

**Configuration**:
```yaml
colored_noise:
  enabled: true
  noise_colors: ['white', 'pink', 'brown']  # Random selection
  zone: [0.0, 1.0]        # Apply to entire audio
  coverage: 1.0           # 100% of zone
```

**Effect**: Adds realistic background noise (white/pink/brown)

**Use case**: Simulates noisy call center environment

### 2. Crop Audio (nlpaug)

**Purpose**: Simulate VoIP packet loss

**Configuration**:
```yaml
crop_audio:
  enabled: true
  zone: [0.1, 0.9]        # Avoid cropping start/end
  coverage: 0.15          # Remove 15% of audio
```

**Effect**: Removes ~15% of audio (packet loss)

**Use case**: Simulates network issues in VoIP calls

### 3. VTLP - Vocal Tract Length Perturbation (nlpaug)

**Purpose**: Speaker variability

**Configuration**:
```yaml
vtlp:
  enabled: true
  factor_range: [0.9, 1.1]  # ±10% vocal tract length
  zone: [0.2, 0.8]          # Apply to middle section
  coverage: 0.1             # 10% of zone
```

**Effect**: Modifies vocal tract characteristics (speaker variation)

**Use case**: Increases diversity in speaker characteristics

### 4. Mask Audio (nlpaug)

**Purpose**: Codec artifacts

**Configuration**:
```yaml
mask_audio:
  enabled: true
  zone: [0.1, 0.9]        # Avoid masking start/end
  coverage: 0.1           # Mask 10% of audio
  mask_with_noise: true   # Fill with noise
```

**Effect**: Masks 10% of audio with noise (codec artifacts)

**Use case**: Simulates audio compression artifacts

### 5. Pitch Shift (librosa)

**Purpose**: Pitch variation

**Configuration**:
```yaml
pitch_shift:
  enabled: true
  n_steps_range: [-2, 2]  # ±2 semitones
```

**Effect**: Shifts pitch up or down

**Use case**: Speaker pitch diversity

### 6. Time Stretch (librosa)

**Purpose**: Speed variation

**Configuration**:
```yaml
time_stretch:
  enabled: true
  rate_range: [0.9, 1.1]  # 0.9=slower, 1.1=faster
```

**Effect**: Changes speech rate (90%-110%)

**Use case**: Speaking speed diversity

### 7. Volume Variation (librosa)

**Purpose**: Volume level changes

**Configuration**:
```yaml
volume_variation:
  enabled: true
  gain_db_range: [-6, 6]  # ±6 dB
```

**Effect**: Adjusts volume level

**Use case**: Simulates varying microphone distances

## Usage Examples

### Basic Usage

```python
from audio_augmenter import AudioAugmenter
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize augmenter
augmenter = AudioAugmenter(config)

# Load audio
audio, sr = augmenter.load_audio("input.wav")

# Select techniques (probability-weighted)
techniques = augmenter.select_augmentation_techniques()
print(f"Selected: {techniques}")

# Apply augmentations
augmented = augmenter.apply_augmentation(audio, sr, techniques)

# Save result
augmenter.save_audio(augmented, sr, "output.wav")
```

### Testing Augmentations

```bash
# Run test script
python test_augmentation.py
```

Expected output:
```
Testing nlpaug augmenters:
  ✓ colored_noise: (94144,)
  ✓ crop_audio: (82847,)
  ✓ vtlp: (94128,)
  ✓ mask_audio: (94144,)

Testing full augmentation pipeline:
  ✓ Test 1 - Techniques: ['crop_audio', 'vtlp'], Output: (82485,)
  ✓ All tests passed!
```

### Dataset Augmentation

```python
import pandas as pd
from pathlib import Path

# Load dataset
df = pd.read_csv("dataset.csv")

# Augment dataset
augmenter = AudioAugmenter(config)
df_augmented, stats = augmenter.augment_dataset(
    df,
    output_dir=Path("output")
)

print(f"Original: {stats['original_samples']}")
print(f"Augmented: {stats['augmented_samples']}")
print(f"Total: {stats['total_samples']}")
print(f"Techniques used: {stats['techniques_applied']}")
```

## Migration from Legacy Approach

### Old Configuration (Strategy-Based)

```yaml
augmentation:
  strategy: "random"  # or "all", "single"
  min_techniques_per_sample: 1
  max_techniques_per_sample: 2
```

### New Configuration (Probability-Weighted)

```yaml
augmentation:
  probabilities:
    colored_noise: 0.8
    crop_audio: 0.4
    # ... etc
```

**Why migrate?**
- More control over technique distribution
- Better suited for call center ASR optimization
- Independent technique selection (not mutually exclusive)

**Backward compatibility**: Legacy strategy-based selection still works if `probabilities` is not set.

## Call Center ASR Optimization

### Recommended Probabilities

Based on common call center challenges:

```yaml
probabilities:
  # High priority (most common issues)
  colored_noise: 0.8       # Background noise very common
  volume_variation: 0.7    # Varying microphone distances

  # Medium priority
  vtlp: 0.5                # Speaker diversity
  crop_audio: 0.4          # Packet loss in VoIP

  # Low priority
  mask_audio: 0.3          # Codec artifacts
  pitch_shift: 0.3         # Natural pitch variation
  time_stretch: 0.3        # Speaking speed variation
```

### Sample Rate

All augmentations maintain **16kHz sample rate** for Whisper compatibility.

## Technical Details

### Array Handling

nlpaug augmenters return lists with extra dimensions. The pipeline automatically:
1. Converts lists to numpy arrays
2. Squeezes extra dimensions
3. Maintains proper shape for chaining

```python
# Handled automatically in apply_augmentation()
augmented = nlpaug_augmenter.augment(audio)
if isinstance(augmented, list):
    augmented = np.array(augmented)
if augmented.ndim > 1:
    augmented = np.squeeze(augmented)
```

### Technique Chaining

Techniques are applied sequentially:

```
Input audio (94144,)
  ↓ colored_noise
  → (94144,)
  ↓ crop_audio
  → (82847,)  # reduced by 15%
  ↓ pitch_shift
  → (82847,)
  ↓ volume_variation
Output (82847,)
```

### Error Handling

- Failed techniques are logged but don't stop the pipeline
- Fallback: If no techniques selected, picks one random enabled technique
- Invalid audio files are skipped with error logging

## Troubleshooting

### Issue: "NoiseAug got unexpected keyword argument 'sampling_rate'"

**Solution**: nlpaug 1.1.11's NoiseAug doesn't accept `sampling_rate`. This is already fixed in `audio_augmenter.py`.

### Issue: "'list' object has no attribute 'shape'"

**Solution**: nlpaug returns lists. This is automatically handled by array conversion in `apply_augmentation()`.

### Issue: Extra dimension in output (1, 94144) instead of (94144,)

**Solution**: Use `np.squeeze()` after nlpaug augmentation. This is automatically handled.

### Issue: Different output length after augmentation

**Expected behavior**:
- `crop_audio` reduces length (packet loss)
- `vtlp` may slightly change length (resampling)
- Other techniques preserve length

## Performance Tips

1. **Use probability weights wisely**: High probabilities = more augmented data = longer processing
2. **Augmentation factor**: `factor=3` triples processing time
3. **Disable unused techniques**: Set `enabled: false` to skip initialization
4. **Batch processing**: Use `augment_dataset()` for efficient batch processing

## References

- [nlpaug Documentation](https://github.com/makcedward/nlpaug)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Whisper ASR](https://github.com/openai/whisper)

## Version History

- **v2.0** (2025-11-05): Hybrid nlpaug + librosa approach with probability weights
- **v1.0** (Initial): librosa-only with strategy-based selection
