#!/usr/bin/env bash
# Preprocess ShapeNet OBJ meshes into SDF samples using compute_sdf.py.
# Input:  /mnt/tim/data/ShapeNetCore/03001627_objs/03001627  (one .obj per subfolder)
# Output: /mnt/tim/data/ShapeNetCore/03001627_sdf            (one .npz per model)
# Log:    /mnt/tim/text2objectsdf/logs/preprocessing_03001627.log  (monitor with: tail -f ...)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/mnt/tim/data/ShapeNetCore/03001627_objs/03001627"
OUTPUT_DIR="/mnt/tim/data/ShapeNetCore/03001627_sdf"
LOG_DIR="/mnt/tim/text2objectsdf/logs"
LOG_FILE="${LOG_DIR}/preprocessing_03001627.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "Preprocessing: OBJ -> SDF"
echo "  Input:   ${INPUT_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Log:     ${LOG_FILE}"
echo "  Started: $(date -Iseconds)"
echo ""

mkdir -p "${OUTPUT_DIR}"
cd "${SCRIPT_DIR}"
python sampling/compute_sdf.py \
  --input-dir "${INPUT_DIR}/models" \
  --output-dir "${OUTPUT_DIR}"

echo ""
echo "Preprocessing finished at $(date -Iseconds)."
