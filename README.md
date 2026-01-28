# ASR Data Augmentation Pipeline

A configurable pipeline for converting Label Studio ASR exports into Whisper-ready datasets with audio augmentation capabilities.

## Features

- **Data Cleaning**: Remove duplicates, filter unnecessary columns, validate data quality
- **Audio Management**: Download audio files from remote EC2 server, remap paths
- **Smart Splitting**: Configurable train/validation/test splits (default 80/10/10)
- **Audio Augmentation**: Multiple techniques with configurable strategies
  - Pitch shifting
  - Time stretching
  - Volume variation
  - Background noise injection
- **Flexible Configuration**: YAML-based configuration for all parameters
- **MLflow Integration**: Automatic experiment tracking and logging
- **Reusable**: Run on multiple Label Studio exports with minimal changes

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/BITZ-IT-Consulting-LTD/asr-data-augmentation-pipeline.git
cd asr-data-augmentation-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### 2. Configuration

Copy and edit the example configuration:

```bash
cp examples/example_config.yaml config/config.yaml
```

Edit `config/config.yaml` to match your setup:

```yaml
input:
  csv_path: "your-label-studio-export.csv"
  audio_remote_path: "user@host:/path/to/audio/"
  ssh_key_path: "~/your-key.pem"

output:
  dataset_name: "your_dataset_name"

augmentation:
  enabled: true
  factor: 3  # 3x the training data
```

### 3. Run Pipeline

```bash
# Run the complete pipeline
python -m asr_pipeline.pipeline

# Or with custom config
python -m asr_pipeline.pipeline --config custom_config.yaml

# Show current configuration
python -m asr_pipeline.pipeline --show-config

# If installed as package, you can also use:
asr-pipeline --config config/config.yaml
```

### 4. Download Audio Files

After running the pipeline, download the audio files:

```bash
bash scripts/download_audio.sh
```

**What this does:**
- Downloads **only the 3,573 specific files** you need (not all 7,041 files in the EC2 folder)
- Uses efficient `rsync --files-from` for selective downloading
- Can be safely interrupted and re-run (rsync resumes from where it left off)
- Verifies all files were downloaded correctly

### 5. Use the Dataset

The pipeline generates TSV files ready for Whisper fine-tuning:

```
output/your_dataset_name/
├── train.tsv          # Training set (augmented)
├── val.tsv            # Validation set
├── test.tsv           # Test set
├── audio/             # Audio files directory
│   └── augmented/     # Augmented audio files
└── stats.json         # Pipeline statistics
```

## Pipeline Workflow

```
Label Studio CSV → Cleaning → Splitting → Augmentation → TSV Files
                      ↓          ↓           ↓              ↓
                   Dedup     80/10/10   Pitch/Time/    train.tsv
                   Filter               Volume/Noise    val.tsv
                                                        test.tsv
```

## Configuration Guide

### Input Configuration

```yaml
input:
  csv_path: "project-3-export.csv"  # Label Studio export
  audio_remote_path: "ec2-user@host:/opt/label-studio/media/project_3/"
  ssh_key_path: "~/openchs.pem"
  label_studio_prefix: "/data/media/project_3/"  # Path in CSV
  local_audio_dir: "output/audio/"  # Local audio directory
```

### Data Cleaning

```yaml
cleaning:
  remove_duplicates: true
  duplicate_strategy: "keep_first"  # or "keep_last"
  keep_columns:
    - audio
    - transcription
    - duration
```

### Dataset Splitting

```yaml
splitting:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  shuffle: true
```

### Augmentation Configuration

```yaml
augmentation:
  enabled: true
  apply_to: "train"  # train, val, test, or all
  factor: 3  # Dataset size multiplier (1 = no augmentation)

  # Strategy options:
  # - "random": each sample gets random combination
  # - "all": each sample gets all techniques
  # - "single": each sample gets exactly one technique
  strategy: "random"

  min_techniques_per_sample: 1
  max_techniques_per_sample: 2

  techniques:
    pitch_shift:
      enabled: true
      n_steps_range: [-2, 2]  # semitones

    time_stretch:
      enabled: true
      rate_range: [0.9, 1.1]  # speed factor

    volume_variation:
      enabled: true
      gain_db_range: [-6, 6]  # decibels

    background_noise:
      enabled: true
      snr_db_range: [15, 25]  # signal-to-noise ratio
```

## Augmentation Strategies

### Strategy: Random (Recommended)
- Each sample gets 1-2 random techniques
- Best balance of diversity and dataset size
- Example: Sample A gets pitch+volume, Sample B gets time stretch only

```yaml
strategy: "random"
min_techniques_per_sample: 1
max_techniques_per_sample: 2
factor: 3  # 3x original size
```

**Result**: Training set becomes ~3x original size with good variety

### Strategy: All
- Each enabled technique creates one augmented version per sample
- Maximum diversity but larger dataset
- Example: 4 techniques = 5x dataset (original + 4 augmented)

```yaml
strategy: "all"
factor: 5  # Original + 4 techniques
```

**Result**: Training set becomes 5x original size

### Strategy: Single
- Each sample gets exactly one random technique
- Minimal augmentation for smaller datasets

```yaml
strategy: "single"
factor: 2  # 2x original size
```

**Result**: Training set becomes 2x original size

## Output Format

### TSV File Format

```
path	transcription	duration
output/audio/sample1.wav	Neno lake ni nzuri	5.2
output/audio/sample1_aug1.wav	Neno lake ni nzuri	5.2
output/audio/sample2.wav	Habari za asubuhi	4.8
```

### Statistics JSON

```json
{
  "cleaning": {
    "initial_count": 3676,
    "duplicate_count": 103,
    "removed_count": 103,
    "final_count": 3573
  },
  "splitting": {
    "train_samples": 2858,
    "val_samples": 357,
    "test_samples": 358
  },
  "augmentation": {
    "original_samples": 2858,
    "augmented_samples": 5716,
    "total_samples": 8574,
    "actual_factor": 3.0
  }
}
```

## Reusing the Pipeline

To process a new Label Studio export:

1. Create a new config file:
```bash
cp config/config.yaml config/config_new_dataset.yaml
```

2. Update the input section:
```yaml
input:
  csv_path: "new-export.csv"

output:
  dataset_name: "new_dataset_v1"
```

3. Run the pipeline:
```bash
python -m asr_pipeline.pipeline --config config/config_new_dataset.yaml
```

## Module Usage

Each module can be used independently:

### Data Cleaner
```python
from asr_pipeline import DataCleaner

cleaner = DataCleaner(config)
df_clean, stats = cleaner.clean("export.csv")
```

### Data Splitter
```python
from asr_pipeline import DataSplitter

splitter = DataSplitter(config)
train_df, val_df, test_df, stats = splitter.split_dataset(df)
```

### Audio Augmenter
```python
from asr_pipeline import AudioAugmenter

augmenter = AudioAugmenter(config)
df_augmented, stats = augmenter.augment_dataset(train_df, output_dir)
```

## MLflow Tracking

The pipeline automatically logs to MLflow:

```bash
# View experiments
mlflow ui

# Then open http://localhost:5000
```

Logged metrics:
- Sample counts (initial, cleaned, split, augmented)
- Augmentation factor
- Pipeline duration
- Configuration parameters

## Troubleshooting

### Audio files not found
**Solution**: Run the download script first:
```bash
bash output/download_audio.sh
```

### Augmentation takes too long
**Solution**: Reduce augmentation factor or disable some techniques:
```yaml
augmentation:
  factor: 2  # Instead of 3
  techniques:
    pitch_shift:
      enabled: false  # Disable slow techniques
```

### Out of memory during augmentation
**Solution**: Process in batches by splitting the CSV:
```python
# Split CSV into smaller files
df = pd.read_csv("large_export.csv")
for i, chunk in enumerate(np.array_split(df, 5)):
    chunk.to_csv(f"export_part_{i}.csv", index=False)
```

### Different Label Studio export format
**Solution**: Update the `keep_columns` in config to match your CSV structure:
```yaml
cleaning:
  keep_columns:
    - your_audio_column
    - your_text_column
    - your_duration_column
```

## Project Structure

```
asr-data-augmentation-pipeline/
├── LICENSE                  # GPL-3.0 License
├── README.md               # This file
├── CLAUDE.md               # Claude Code guidance
├── setup.py                # Package installation
├── requirements.txt        # Python dependencies
├── src/
│   └── asr_pipeline/       # Main package
│       ├── __init__.py
│       ├── pipeline.py     # Main orchestrator
│       ├── audio_augmenter.py
│       ├── audio_downloader.py
│       ├── data_cleaner.py
│       └── data_splitter.py
├── config/
│   └── config.yaml         # Configuration template
├── scripts/
│   ├── quick_start.sh      # Setup script
│   └── export_dataset.sh   # Dataset export script
├── tests/
│   └── test_augmentation.py
├── docs/
│   ├── AUGMENTATION_GUIDE.md
│   └── DATASET_EXPORT_GUIDE.md
├── examples/
│   └── example_config.yaml
└── output/                 # Generated outputs (gitignored)
    ├── download_audio.sh   # Generated download script
    ├── pipeline.log        # Pipeline logs
    └── dataset_name/
        ├── train.tsv
        ├── val.tsv
        ├── test.tsv
        ├── stats.json
        └── audio/
            └── augmented/
```

## Advanced Usage

### Custom Augmentation Parameters

Fine-tune augmentation for your use case:

```yaml
# For telephone/phone audio
techniques:
  pitch_shift:
    n_steps_range: [-1, 1]  # Smaller range
  background_noise:
    snr_db_range: [10, 15]  # More noise

# For clean studio recordings
techniques:
  pitch_shift:
    n_steps_range: [-3, 3]  # Larger range
  background_noise:
    snr_db_range: [20, 30]  # Less noise
```

### Selective Augmentation

Only augment samples longer than X seconds:

```python
# Modify src/asr_pipeline/audio_augmenter.py augment_dataset method
if row['duration'] < 4.0:
    continue  # Skip short samples
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{asr_data_augmentation_pipeline,
  author = {k_nurf},
  title = {ASR Data Augmentation Pipeline},
  year = {2025},
  url = {https://github.com/BITZ-IT-Consulting-LTD/asr-data-augmentation-pipeline}
}
```

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [documentation](docs/)
3. Check logs in `output/pipeline.log`
4. Open an [issue](https://github.com/BITZ-IT-Consulting-LTD/asr-data-augmentation-pipeline/issues)
