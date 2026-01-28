# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASR Data Augmentation Pipeline for converting Label Studio exports into Whisper-ready datasets with audio augmentation capabilities. Designed for call center ASR with hybrid nlpaug + librosa augmentation techniques.

**Key capabilities:**
- Data cleaning (deduplication, filtering, validation)
- Train/validation/test splitting (default 70/20/10)
- Audio augmentation with 7 techniques (colored noise, pitch shift, time stretch, volume variation, VTLP, crop audio, mask audio)
- MLflow experiment tracking integration
- SSH-based audio download from remote servers

## Development Commands

### Pipeline Execution

```bash
# Run complete pipeline with default config
python -m asr_pipeline.pipeline

# Or if installed as package
asr-pipeline

# Run with custom config
python -m asr_pipeline.pipeline --config config/custom_config.yaml

# Show current configuration without running
python -m asr_pipeline.pipeline --show-config
```

### Testing

```bash
# Test augmentation techniques
python tests/test_augmentation.py

# Expected output shows successful initialization of all augmenters:
# ✓ colored_noise, ✓ crop_audio, ✓ vtlp, ✓ mask_audio
```

### Audio Download

```bash
# After pipeline generates download script, fetch audio files
bash output/download_audio.sh

# Uses rsync --files-from for selective downloading
# Can be safely interrupted and resumed
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .

# Quick start script (automated setup)
bash scripts/quick_start.sh
```

### MLflow Tracking

```bash
# View experiments (if MLflow enabled in config)
mlflow ui

# Opens at http://localhost:5000 (or configured tracking_uri)
```

## Architecture

### Pipeline Flow

The pipeline orchestrates five modules in sequence:

```
Label Studio CSV
    ↓
[1] DataCleaner → Remove duplicates, filter columns, validate
    ↓
[2] AudioDownloader → Remap paths, generate rsync script
    ↓
[3] DataSplitter → Split into train/val/test sets
    ↓
[4] AudioAugmenter → Apply augmentation techniques (training set)
    ↓
[5] TSV Output → Save train.tsv, val.tsv, test.tsv
```

Each module is independent and can be imported/used separately.

### Configuration-Driven Design

All pipeline behavior is controlled via YAML configuration (`config.yaml`):

- **Input**: CSV paths, SSH credentials, remote audio paths
- **Cleaning**: Duplicate strategy, column filtering
- **Splitting**: Train/val/test ratios, random seed
- **Augmentation**: Enabled techniques, probability weights, factor multiplier
- **MLflow**: Experiment name, tracking URI
- **Logging**: Level, output file

**Key principle**: Pipeline must be fully reproducible from config alone.

### Augmentation System Architecture

Hybrid approach combining nlpaug (call center-specific) and librosa (proven techniques):

**nlpaug augmenters** (call center environment simulation):
- `colored_noise`: Background noise (white/pink/brown)
- `crop_audio`: VoIP packet loss simulation
- `vtlp`: Speaker variability (vocal tract length perturbation)
- `mask_audio`: Codec artifacts

**librosa augmenters** (standard transformations):
- `pitch_shift`: Pitch variation (±2 semitones)
- `time_stretch`: Speed variation (0.9-1.1x)
- `volume_variation`: Volume changes (±6 dB)

**Probability-weighted selection**: Each technique is independently applied based on configured probability (e.g., `colored_noise: 0.5` = 50% chance). This replaces legacy strategy-based selection (`random`, `all`, `single`).

**Augmentation factor**: Controls dataset size multiplier (e.g., `factor: 3` = 3x original size).

**Array handling**: nlpaug returns lists with extra dimensions. `AudioAugmenter.apply_augmentation()` automatically converts to numpy arrays and squeezes dimensions for proper chaining.

**Sample rate**: All augmentations maintain 16kHz for Whisper compatibility.

### Module Responsibilities

**`src/asr_pipeline/pipeline.py`** (orchestrator):
- Coordinates all modules
- Sets up logging and MLflow
- Generates statistics and reports
- Entry point for pipeline execution

**`src/asr_pipeline/data_cleaner.py`**:
- Loads Label Studio CSV exports
- Removes duplicates (configurable strategy: keep_first/keep_last)
- Filters columns (keeps only: audio, transcription, duration)
- Returns cleaned DataFrame + statistics

**`src/asr_pipeline/audio_downloader.py`**:
- Remaps Label Studio paths to local paths
- Generates rsync download script with `--files-from` for selective download
- Validates file existence and counts missing files

**`src/asr_pipeline/data_splitter.py`**:
- Splits dataset into train/val/test with stratification
- Shuffles data with reproducible random seed
- Saves TSV files in Whisper format (path, transcription, duration)

**`src/asr_pipeline/audio_augmenter.py`**:
- Initializes nlpaug and librosa augmenters from config
- Selects techniques using probability weights
- Applies augmentations sequentially (technique chaining)
- Handles array conversions and dimension squeezing
- Saves augmented audio files with `_aug{N}` suffix
- Returns augmented DataFrame + statistics

### Output Structure

```
output/                     # Generated outputs (gitignored)
├── download_audio.sh       # Generated rsync script
├── pipeline.log           # Execution logs
└── {dataset_name}/
    ├── train.tsv          # Training set (augmented)
    ├── val.tsv            # Validation set
    ├── test.tsv           # Test set
    ├── stats.json         # Pipeline statistics
    └── audio/
        └── augmented/     # Augmented audio files
```

### Source Structure

```
src/asr_pipeline/
├── __init__.py            # Package initialization
├── pipeline.py            # Main orchestrator
├── audio_augmenter.py     # Augmentation logic
├── audio_downloader.py    # Audio download management
├── data_cleaner.py        # Data cleaning
└── data_splitter.py       # Dataset splitting
```

## Project Conventions

### MLOps Best Practices

When implementing or modifying pipeline code:

1. **Experiment Tracking**: Use MLflow for all experiments
   - Experiment naming: `task_name_[author]`
   - Log parameters, metrics, and artifacts
   - Set tracking URI in config

2. **Configuration Management**:
   - All hyperparameters in YAML config
   - No hard-coded paths or values
   - Validate configs before execution

3. **Data Versioning**:
   - Use DVC for datasets and models
   - Never commit large files directly
   - Only commit `.dvc` files

4. **Reproducibility**:
   - Set random seeds consistently
   - Log all configuration parameters
   - Pipeline must be reproducible from config alone

### Code Quality Standards

- **Type hints**: Use type annotations consistently
- **Docstrings**: Document all public functions/classes
- **Modularity**: Keep functions single-purpose and reusable
- **Error handling**: Log errors, don't fail silently
- **Progress tracking**: Use tqdm for long-running operations

### Audio Processing Standards

- **Sample rate**: Maintain 16kHz for Whisper compatibility
- **Normalization**: Validate audio amplitude and duration
- **Augmentation chaining**: Ensure augmentations are composable
- **Determinism**: Always set random seed for reproducibility

## Key Configuration Patterns

### Augmentation Strategy Migration

**Legacy (strategy-based)**:
```yaml
augmentation:
  strategy: "random"
  min_techniques_per_sample: 1
  max_techniques_per_sample: 2
```

**Current (probability-weighted)**:
```yaml
augmentation:
  probabilities:
    colored_noise: 0.5
    volume_variation: 0.7
    # Each technique independently applied
```

Use probability weights for better control over technique distribution.

### Common Configuration Scenarios

**Disable augmentation** (first run before audio download):
```yaml
augmentation:
  enabled: false
```

**Reduce augmentation factor** (faster processing):
```yaml
augmentation:
  factor: 2  # Instead of 3 or 5
```

**Disable slow techniques**:
```yaml
techniques:
  pitch_shift:
    enabled: false
  vtlp:
    enabled: false
```

**Change split ratios**:
```yaml
splitting:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

## Special Agent: mlops-pipeline-auditor

This repository includes a Claude Code agent (`mlops-pipeline-auditor`) for evaluating pipeline code quality.

**When to use**: After implementing/modifying pipeline components, before PRs, during refactoring.

**What it checks**:
- MLOps maturity (MLflow, DVC, config management)
- Code quality (modularity, type hints, documentation)
- Audio processing best practices
- Configuration flexibility
- Performance and efficiency

**Usage**: The agent is automatically available via the Task tool when working in this repository.

## Common Patterns

### Running Multiple Configurations

```bash
# Create config variants
cp config/config.yaml config/config_v1.yaml
cp config/config.yaml config/config_v2.yaml

# Edit configs with different augmentation settings
# Then run each
python -m asr_pipeline.pipeline --config config/config_v1.yaml
python -m asr_pipeline.pipeline --config config/config_v2.yaml
```

### Selective Augmentation

To augment only samples meeting certain criteria, modify [src/asr_pipeline/audio_augmenter.py](src/asr_pipeline/audio_augmenter.py):

```python
# In augment_dataset() method
if row['duration'] < 4.0:
    continue  # Skip short samples
```

### Debugging Failed Augmentations

Check `output/pipeline.log` for detailed error messages:

```bash
tail -f output/pipeline.log  # Watch logs in real-time
grep ERROR output/pipeline.log  # Find errors
```

## Dependencies

Core libraries:
- **pandas, numpy**: Data manipulation
- **librosa**: Audio processing and augmentation
- **soundfile**: Audio I/O
- **nlpaug 1.1.11**: Audio augmentation (call center-specific)
- **scikit-learn**: Data splitting
- **mlflow**: Experiment tracking
- **tqdm**: Progress bars

Development tools:
- **pytest**: Testing
- **black**: Code formatting

## Notes

- **Audio files**: Pipeline generates download script but doesn't auto-download. Run `bash output/download_audio.sh` manually.
- **First run**: Set `augmentation.enabled: false` until audio files are downloaded.
- **Memory**: For large datasets (>10k samples), consider reducing augmentation factor or disabling techniques.
- **nlpaug quirks**: Returns lists with extra dimensions. `AudioAugmenter` handles conversion automatically.
- **Sample rate**: Pipeline assumes 16kHz. If source audio differs, it will be resampled.
