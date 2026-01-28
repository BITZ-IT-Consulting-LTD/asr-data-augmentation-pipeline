#!/bin/bash
# Export Dataset Script
# Creates a portable copy of your ASR dataset

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "ASR Dataset Export Tool"
echo -e "========================================${NC}"
echo ""

# Prompt for dataset name
echo -e "${YELLOW}Enter the dataset name:${NC}"
read -p "(default: swahili_asr_v2): " DATASET_NAME
DATASET_NAME=${DATASET_NAME:-swahili_asr_v2}

# Prompt for output directory
echo ""
echo -e "${YELLOW}Enter the output directory where dataset is located:${NC}"
read -p "(default: output): " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-output}

# Check if dataset exists
if [ ! -d "$OUTPUT_DIR/$DATASET_NAME" ]; then
    echo -e "${YELLOW}Warning: Directory $OUTPUT_DIR/$DATASET_NAME not found!${NC}"
    echo "Available datasets in $OUTPUT_DIR:"
    ls -1 "$OUTPUT_DIR" 2>/dev/null | grep -v "audio\|\.log\|\.txt\|\.sh" || echo "  (none found)"
    echo ""
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo "Export cancelled."
        exit 1
    fi
fi

# Prompt for export directory name
echo ""
echo -e "${YELLOW}Enter the export directory name:${NC}"
read -p "(default: ${DATASET_NAME}_export): " EXPORT_DIR
EXPORT_DIR=${EXPORT_DIR:-${DATASET_NAME}_export}

# Confirm before proceeding
echo ""
echo -e "${BLUE}========================================"
echo "Export Configuration:"
echo -e "========================================${NC}"
echo "  Dataset name: $DATASET_NAME"
echo "  Source directory: $OUTPUT_DIR"
echo "  Export directory: $EXPORT_DIR"
echo "  Archive file: ${EXPORT_DIR}.tar.gz"
echo ""
read -p "Proceed with export? (y/n): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Export cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}========================================"
echo "Exporting ASR Dataset: $DATASET_NAME"
echo -e "========================================${NC}"

# Create export directory
mkdir -p "$EXPORT_DIR"

echo "Copying dataset files..."

# Copy the entire output structure (preserves relative paths)
echo "  - Copying TSV files..."
cp -r "$OUTPUT_DIR/$DATASET_NAME" "$EXPORT_DIR/"

echo "  - Copying original audio files..."
mkdir -p "$EXPORT_DIR/audio"
cp -r "$OUTPUT_DIR/audio/"*.wav "$EXPORT_DIR/audio/" 2>/dev/null || echo "    (No audio files found in output/audio/)"

echo "  - Copying augmented audio files..."
# Augmented files are already in the dataset folder

# Copy stats and documentation
echo "  - Copying metadata..."
cp "$OUTPUT_DIR/$DATASET_NAME/stats.json" "$EXPORT_DIR/$DATASET_NAME/" 2>/dev/null || true

# Fix paths in TSV files to be relative to export directory
echo "  - Fixing paths in TSV files..."
FIXED_COUNT=0
for tsv_file in "$EXPORT_DIR/$DATASET_NAME"/*.tsv; do
    if [ -f "$tsv_file" ]; then
        # Replace output/audio/ with audio/
        sed -i "s|${OUTPUT_DIR}/audio/|audio/|g" "$tsv_file"
        # Replace output/{dataset}/ with {dataset}/
        sed -i "s|${OUTPUT_DIR}/${DATASET_NAME}/|${DATASET_NAME}/|g" "$tsv_file"
        echo "    ✓ Fixed: $(basename "$tsv_file")"
        FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
done
echo "    Fixed paths in $FIXED_COUNT TSV files"

# Verify paths are relative (sample check)
echo "  - Verifying paths..."
SAMPLE_LINE=$(head -2 "$EXPORT_DIR/$DATASET_NAME/train.tsv" 2>/dev/null | tail -1 | cut -f1)
if [[ "$SAMPLE_LINE" == audio/* ]] || [[ "$SAMPLE_LINE" == ${DATASET_NAME}/* ]]; then
    echo -e "    ${GREEN}✓ Paths are relative and ready to use${NC}"
else
    echo -e "    ${YELLOW}⚠ Warning: Paths may not be relative (sample: $SAMPLE_LINE)${NC}"
fi

# Get dataset statistics
TRAIN_COUNT=$(wc -l < "$OUTPUT_DIR/$DATASET_NAME/train.tsv" 2>/dev/null || echo "0")
VAL_COUNT=$(wc -l < "$OUTPUT_DIR/$DATASET_NAME/val.tsv" 2>/dev/null || echo "0")
TEST_COUNT=$(wc -l < "$OUTPUT_DIR/$DATASET_NAME/test.tsv" 2>/dev/null || echo "0")
# Subtract 1 for header line
TRAIN_COUNT=$((TRAIN_COUNT - 1))
VAL_COUNT=$((VAL_COUNT - 1))
TEST_COUNT=$((TEST_COUNT - 1))
TOTAL_COUNT=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))

AUDIO_COUNT=$(find "$OUTPUT_DIR/audio" -name "*.wav" 2>/dev/null | wc -l || echo "0")
AUG_COUNT=$(find "$OUTPUT_DIR/$DATASET_NAME/audio/augmented" -name "*.wav" 2>/dev/null | wc -l || echo "0")

# Create README for the export
cat > "$EXPORT_DIR/README.md" << EOF
# ASR Dataset Export: $DATASET_NAME

## Structure:
${EXPORT_DIR}/
├── audio/                    # Original audio files (${AUDIO_COUNT})
├── ${DATASET_NAME}/
│   ├── train.tsv            # Training set (${TRAIN_COUNT} samples)
│   ├── val.tsv              # Validation set (${VAL_COUNT} samples)
│   ├── test.tsv             # Test set (${TEST_COUNT} samples)
│   ├── stats.json           # Dataset statistics
│   └── audio/
│       └── augmented/       # Augmented audio files (${AUG_COUNT})

## Usage:

### On the new computer:

1. Extract the archive:
   tar -xzf ${EXPORT_DIR}.tar.gz
   cd ${EXPORT_DIR}

2. The dataset is ready to use! All paths in TSV files are relative and pre-configured.

3. For Whisper training, use the TSV files:
   - Training: ${DATASET_NAME}/train.tsv
   - Validation: ${DATASET_NAME}/val.tsv
   - Test: ${DATASET_NAME}/test.tsv

   Example with Hugging Face:
   \`\`\`python
   from datasets import load_dataset

   dataset = load_dataset(
       'csv',
       data_files={
           'train': '${DATASET_NAME}/train.tsv',
           'validation': '${DATASET_NAME}/val.tsv',
           'test': '${DATASET_NAME}/test.tsv'
       },
       delimiter='\\t'
   )
   \`\`\`

## Dataset Info:
- Dataset name: ${DATASET_NAME}
- Total samples: ${TOTAL_COUNT} (${TRAIN_COUNT} train + ${VAL_COUNT} val + ${TEST_COUNT} test)
- Original audio: ${AUDIO_COUNT} files
- Augmented audio: ${AUG_COUNT} files
- Format: WAV files (16kHz mono) + TSV manifests
- Export date: $(date +"%Y-%m-%d %H:%M:%S")

## Notes:
- ✓ All paths in TSV files are relative and ready to use (no manual fixes needed)
- ✓ Audio files maintain 16kHz sample rate for Whisper compatibility
- ✓ Augmented files include various transformations (noise, pitch, speed, etc.)
- ✓ Dataset structure is portable - works on any system after extraction
EOF

# Create archive
echo ""
echo "Creating archive..."
ARCHIVE_NAME="${EXPORT_DIR}.tar.gz"
tar -czf "$ARCHIVE_NAME" "$EXPORT_DIR"

# Calculate sizes
EXPORT_SIZE=$(du -sh "$EXPORT_DIR" | cut -f1)
ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)

echo ""
echo -e "${GREEN}========================================"
echo "Export Complete!"
echo -e "========================================${NC}"
echo "Export directory: $EXPORT_DIR ($EXPORT_SIZE)"
echo "Archive file: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
echo ""
echo "Dataset Statistics:"
echo "  - Total samples: $TOTAL_COUNT"
echo "  - Train: $TRAIN_COUNT | Val: $VAL_COUNT | Test: $TEST_COUNT"
echo "  - Original audio: $AUDIO_COUNT files"
echo "  - Augmented audio: $AUG_COUNT files"
echo ""
echo "To transfer:"
echo "  1. Copy $ARCHIVE_NAME to new location"
echo "  2. Extract: tar -xzf $ARCHIVE_NAME"
echo "  3. Follow instructions in README.md"
echo -e "${GREEN}========================================${NC}"
