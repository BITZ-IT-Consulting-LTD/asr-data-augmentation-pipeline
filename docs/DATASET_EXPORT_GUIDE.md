# Dataset Export & Transfer Guide

This guide explains how to move your ASR dataset to another location or computer.

## Quick Summary

**Files you need:**
- 3,573 original audio files: `output/audio/*.wav`
- 5,714 augmented audio files: `output/swahili_asr_v1/audio/augmented/*.wav`
- 3 TSV files: `output/swahili_asr_v1/{train,val,test}.tsv`
- 1 stats file: `output/swahili_asr_v1/stats.json`

---

## Option 1: Automated Export (Easiest) ⭐

Use the provided export script:

```bash
# Create a portable archive
bash export_dataset.sh

# This creates:
# - swahili_asr_v1_export/ folder
# - swahili_asr_v1_export.tar.gz archive
```

**Transfer to new location:**
```bash
# 1. Copy the archive
scp swahili_asr_v1_export.tar.gz user@newserver:/path/to/destination/

# 2. Extract on new machine
tar -xzf swahili_asr_v1_export.tar.gz
cd swahili_asr_v1_export

# 3. Fix paths in TSV files (if needed)
sed -i 's|output/audio/|audio/|g' swahili_asr_v1/*.tsv
sed -i 's|output/swahili_asr_v1/|swahili_asr_v1/|g' swahili_asr_v1/*.tsv
```

---

## Option 2: Manual Copy

### Step 1: Copy Files

```bash
# Create export directory
mkdir -p swahili_asr_v1_portable/audio
mkdir -p swahili_asr_v1_portable/swahili_asr_v1/audio/augmented

# Copy original audio
cp output/audio/*.wav swahili_asr_v1_portable/audio/

# Copy augmented audio
cp output/swahili_asr_v1/audio/augmented/*.wav swahili_asr_v1_portable/swahili_asr_v1/audio/augmented/

# Copy TSV files
cp output/swahili_asr_v1/*.tsv swahili_asr_v1_portable/swahili_asr_v1/
cp output/swahili_asr_v1/stats.json swahili_asr_v1_portable/swahili_asr_v1/
```

### Step 2: Update Paths

The TSV files contain paths like `output/audio/file.wav`. You need to update these:

```bash
cd swahili_asr_v1_portable

# Update all TSV files
for tsv in swahili_asr_v1/*.tsv; do
    sed -i 's|output/audio/|audio/|g' "$tsv"
    sed -i 's|output/swahili_asr_v1/|swahili_asr_v1/|g' "$tsv"
done
```

### Step 3: Verify

```bash
# Check that paths are correct
head swahili_asr_v1/train.tsv

# Should show paths like:
# audio/filename.wav
# swahili_asr_v1/audio/augmented/filename_aug1.wav
```

---

## Option 3: Restructure for Absolute Paths

If you want to use absolute paths (e.g., for cloud storage):

### Step 1: Copy to Final Location

```bash
# Copy to final destination
mkdir -p /data/asr_datasets/swahili_v1
cp -r output/audio /data/asr_datasets/swahili_v1/
cp -r output/swahili_asr_v1 /data/asr_datasets/swahili_v1/
```

### Step 2: Update TSV with Absolute Paths

```bash
cd /data/asr_datasets/swahili_v1

# Replace relative paths with absolute paths
for tsv in swahili_asr_v1/*.tsv; do
    sed -i "s|output/audio/|/data/asr_datasets/swahili_v1/audio/|g" "$tsv"
    sed -i "s|output/swahili_asr_v1/|/data/asr_datasets/swahili_v1/swahili_asr_v1/|g" "$tsv"
done
```

---

## Option 4: Cloud Storage / S3

For cloud storage, you can upload and update paths to URLs:

```bash
# Upload to S3 (example)
aws s3 sync output/audio/ s3://my-bucket/datasets/swahili_v1/audio/
aws s3 sync output/swahili_asr_v1/audio/augmented/ s3://my-bucket/datasets/swahili_v1/augmented/

# Update TSV files with S3 URLs
sed -i 's|output/audio/|s3://my-bucket/datasets/swahili_v1/audio/|g' swahili_asr_v1/*.tsv
sed -i 's|output/swahili_asr_v1/audio/augmented/|s3://my-bucket/datasets/swahili_v1/augmented/|g' swahili_asr_v1/*.tsv
```

---

## Verification Checklist

After transferring, verify your dataset:

```bash
# Check file counts
echo "Original audio files:"
ls audio/*.wav | wc -l  # Should be 3,573

echo "Augmented audio files:"
ls swahili_asr_v1/audio/augmented/*.wav | wc -l  # Should be 5,714

echo "TSV line counts:"
wc -l swahili_asr_v1/*.tsv
# train.tsv: 8,572 lines (8,571 samples + 1 header)
# val.tsv: 359 lines (358 samples + 1 header)
# test.tsv: 359 lines (358 samples + 1 header)

# Test if paths are valid (check first 5 entries)
head -6 swahili_asr_v1/train.tsv | tail -5 | cut -f1 | while read path; do
    if [ -f "$path" ]; then
        echo "✓ $path exists"
    else
        echo "✗ $path MISSING"
    fi
done
```

---

## Dataset Size

- **Audio files**: ~1.2 GB
  - Original: ~600 MB (3,573 files)
  - Augmented: ~600 MB (5,714 files)
- **TSV files**: ~2 MB
- **Total**: ~1.2 GB

---

## For Whisper Training

Once transferred, use the TSV files directly:

```python
# Example Whisper training code
from datasets import load_dataset

# Load from TSV
train_dataset = load_dataset(
    'csv',
    data_files='swahili_asr_v1/train.tsv',
    delimiter='\t',
    column_names=['audio', 'transcription', 'duration']
)

val_dataset = load_dataset(
    'csv',
    data_files='swahili_asr_v1/val.tsv',
    delimiter='\t',
    column_names=['audio', 'transcription', 'duration']
)
```

Or if paths need to be adjusted in your training code:

```python
import pandas as pd

# Load TSV
df = pd.read_csv('swahili_asr_v1/train.tsv', sep='\t')

# Adjust paths if needed
df['audio'] = df['audio'].apply(lambda x: os.path.join(BASE_PATH, x))
```

---

## Troubleshooting

### Issue: "File not found" errors

**Cause**: TSV paths don't match actual file locations

**Solution**: Update paths in TSV files
```bash
# Check what paths look like
head -2 swahili_asr_v1/train.tsv

# Update accordingly
sed -i 's|old_path|new_path|g' swahili_asr_v1/*.tsv
```

### Issue: Missing audio files

**Cause**: Not all files were copied

**Solution**: Verify file counts
```bash
# Should have 3,573 + 5,714 = 9,287 total audio files
find . -name "*.wav" | wc -l
```

### Issue: Paths work but training fails

**Cause**: Audio files might be corrupted during transfer

**Solution**: Verify a sample of audio files
```bash
# Test 10 random files
find audio -name "*.wav" | shuf -n 10 | while read f; do
    file "$f"
    sox "$f" -n stat 2>&1 | grep Length
done
```

---

## Best Practices

1. **Always verify** file counts after transfer
2. **Test with a subset** before full training run
3. **Keep original** dataset until new location is verified
4. **Document the path structure** used in your setup
5. **Use version control** for TSV files (they're small text files)

---

## Quick Transfer Commands

**To another server:**
```bash
rsync -avz --progress output/ user@server:/path/to/destination/output/
```

**To external drive:**
```bash
cp -r output/ /mnt/external_drive/swahili_asr_backup/
```

**Create compressed archive:**
```bash
tar -czf swahili_asr_v1_$(date +%Y%m%d).tar.gz output/audio output/swahili_asr_v1
```
