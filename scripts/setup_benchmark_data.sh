#!/usr/bin/env bash
# ============================================================
# setup_benchmark_data.sh
#
# Downloads benchmark datasets from GCS to the local repo.
# Run this after cloning the repo on a new server.
#
# Usage:
#   ./scripts/setup_benchmark_data.sh              # download all
#   ./scripts/setup_benchmark_data.sh sd_cassi      # download only sd_cassi
#   ./scripts/setup_benchmark_data.sh cacti         # download only cacti
#   ./scripts/setup_benchmark_data.sh --challenge   # download only challenge HDF5 files
#   ./scripts/setup_benchmark_data.sh --checkpoints  # download pretrained model weights
#   ./scripts/setup_benchmark_data.sh --list        # list available datasets
#
# Prerequisites:
#   - gsutil (Google Cloud SDK) installed and authenticated
#   - Access to gs://pwm-benchmark-datasets bucket
#
# To authenticate on a new server:
#   gcloud auth login
#   gcloud config set project <your-project-id>
# ============================================================

set -euo pipefail

GCS_BUCKET="gs://pwm-benchmark-datasets"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCHMARK_DIR="${REPO_ROOT}/datasets/benchmark"
CHALLENGE_DIR="${REPO_ROOT}/platform/pwm_platform/static/benchmark-data/challenge-data/v1.0"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoint"

# Available benchmark datasets
DATASETS=(sd_cassi cacti spc_kronecker)

# Available model checkpoints (GPU algorithms)
CHECKPOINTS=(DRUNet DnCNN ELP-Unfolding EfficientSCI HATNet-SPI ISTA-Net MST MST-HDNet PnP-CASSI PnP-SCI ProxUnroll)

usage() {
    echo "Usage: $0 [OPTIONS] [DATASET...]"
    echo ""
    echo "Downloads benchmark datasets from GCS."
    echo ""
    echo "Datasets: ${DATASETS[*]}"
    echo ""
    echo "Options:"
    echo "  --challenge      Download only challenge HDF5 files (no source images)"
    echo "  --checkpoints    Download pretrained model weights (17 GB total)"
    echo "  --checkpoint X   Download a specific checkpoint (e.g. ELP-Unfolding)"
    echo "  --all            Download everything (datasets + checkpoints)"
    echo "  --list           List available datasets, checkpoints, and sizes"
    echo "  --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                          # Download all datasets (source + challenge)"
    echo "  $0 sd_cassi cacti           # Download specific datasets"
    echo "  $0 --challenge              # Download only challenge HDF5 files"
    echo "  $0 --challenge cacti        # Download only cacti challenge files"
    echo "  $0 --checkpoints            # Download all model weights"
    echo "  $0 --checkpoint ELP-Unfolding  # Download one checkpoint"
    echo "  $0 --all                    # Download datasets + checkpoints"
}

list_datasets() {
    echo "Available benchmark datasets on GCS:"
    echo ""
    for ds in "${DATASETS[@]}"; do
        echo "  ${ds}:"
        echo "    Source data: ${GCS_BUCKET}/datasets/${ds}/"
        gsutil du -sh "${GCS_BUCKET}/datasets/${ds}/" 2>/dev/null || echo "    (not available)"
        echo "    Challenge files:"
        for tier in public dev hidden; do
            size=$(gsutil du -s "${GCS_BUCKET}/challenge-data/v1.0/${ds}_challenge_${tier}.h5" 2>/dev/null | awk '{print $1}')
            if [ -n "$size" ]; then
                echo "      ${tier}: $(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size} bytes")"
            fi
        done
        echo ""
    done

    echo "Available model checkpoints on GCS:"
    echo ""
    for ckpt in "${CHECKPOINTS[@]}"; do
        size=$(gsutil du -sh "${GCS_BUCKET}/checkpoint/${ckpt}/" 2>/dev/null | head -1 | awk '{print $1" "$2}')
        echo "  ${ckpt}: ${size:-not available}"
    done
    echo ""
    gsutil du -sh "${GCS_BUCKET}/checkpoint/" 2>/dev/null | head -1 | awk '{print "  Total checkpoints: "$1" "$2}'
}

download_source() {
    local ds="$1"
    echo "==> Downloading source data: ${ds}"
    mkdir -p "${BENCHMARK_DIR}/${ds}"
    gsutil -m rsync -r "${GCS_BUCKET}/datasets/${ds}/" "${BENCHMARK_DIR}/${ds}/"
    echo "    Done: ${BENCHMARK_DIR}/${ds}/"
}

download_challenge() {
    local ds="$1"
    echo "==> Downloading challenge HDF5 files: ${ds}"
    mkdir -p "${BENCHMARK_DIR}/${ds}/public" "${BENCHMARK_DIR}/${ds}/dev" "${BENCHMARK_DIR}/${ds}/hidden"
    for tier in public dev hidden; do
        local src="${GCS_BUCKET}/challenge-data/v1.0/${ds}_challenge_${tier}.h5"
        local dst="${BENCHMARK_DIR}/${ds}/${tier}/${ds}_challenge_${tier}.h5"
        if gsutil -q stat "$src" 2>/dev/null; then
            echo "    ${tier}..."
            gsutil cp "$src" "$dst"
        else
            echo "    ${tier}: not found on GCS, skipping"
        fi
    done
    echo "    Done."
}

download_checkpoints() {
    local target="$1"  # "all" or specific checkpoint name
    if [ "$target" = "all" ]; then
        echo "==> Downloading all model checkpoints (17 GB)..."
        mkdir -p "${CHECKPOINT_DIR}"
        gsutil -m rsync -r "${GCS_BUCKET}/checkpoint/" "${CHECKPOINT_DIR}/"
        echo "    Done: ${CHECKPOINT_DIR}/"
    else
        echo "==> Downloading checkpoint: ${target}"
        mkdir -p "${CHECKPOINT_DIR}/${target}"
        gsutil -m rsync -r "${GCS_BUCKET}/checkpoint/${target}/" "${CHECKPOINT_DIR}/${target}/"
        echo "    Done: ${CHECKPOINT_DIR}/${target}/"
    fi
}

# Parse arguments
CHALLENGE_ONLY=false
DOWNLOAD_CHECKPOINTS=false
DOWNLOAD_ALL=false
CHECKPOINT_TARGET=""
SELECTED=()

for arg in "$@"; do
    case "$arg" in
        --challenge) CHALLENGE_ONLY=true ;;
        --checkpoints) DOWNLOAD_CHECKPOINTS=true ;;
        --all) DOWNLOAD_ALL=true ;;
        --list) list_datasets; exit 0 ;;
        --help|-h) usage; exit 0 ;;
        --checkpoint)
            DOWNLOAD_CHECKPOINTS=true
            # Next arg will be captured as checkpoint name
            ;;
        *)
            if [ "$DOWNLOAD_CHECKPOINTS" = true ] && [ -z "$CHECKPOINT_TARGET" ] && [[ " ${CHECKPOINTS[*]} " =~ " ${arg} " ]]; then
                CHECKPOINT_TARGET="$arg"
            else
                SELECTED+=("$arg")
            fi
            ;;
    esac
done

# Verify gsutil is available
if ! command -v gsutil &>/dev/null; then
    echo "ERROR: gsutil not found. Install Google Cloud SDK first:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "PWM Benchmark Data Setup"
echo "========================"

# Handle --checkpoints only mode
if [ "$DOWNLOAD_CHECKPOINTS" = true ] && [ "$DOWNLOAD_ALL" = false ] && [ ${#SELECTED[@]} -eq 0 ] && [ "$CHALLENGE_ONLY" = false ]; then
    if [ -n "$CHECKPOINT_TARGET" ]; then
        download_checkpoints "$CHECKPOINT_TARGET"
    else
        download_checkpoints "all"
    fi
    echo ""
    echo "All done! Checkpoints ready at: ${CHECKPOINT_DIR}"
    exit 0
fi

# Default: all datasets if none specified
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("${DATASETS[@]}")
fi

echo "Target: ${BENCHMARK_DIR}"
echo "Datasets: ${SELECTED[*]}"
echo "Mode: $([ "$CHALLENGE_ONLY" = true ] && echo 'challenge files only' || echo 'full (source + challenge)')"
echo ""

for ds in "${SELECTED[@]}"; do
    if [[ ! " ${DATASETS[*]} " =~ " ${ds} " ]]; then
        echo "WARNING: Unknown dataset '${ds}', skipping. Available: ${DATASETS[*]}"
        continue
    fi

    if [ "$CHALLENGE_ONLY" = true ]; then
        download_challenge "$ds"
    else
        download_source "$ds"
    fi
    echo ""
done

# Download checkpoints if --all or --checkpoints was passed alongside datasets
if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_CHECKPOINTS" = true ]; then
    if [ -n "$CHECKPOINT_TARGET" ]; then
        download_checkpoints "$CHECKPOINT_TARGET"
    else
        download_checkpoints "all"
    fi
    echo ""
fi

echo "All done! Benchmark data is ready at: ${BENCHMARK_DIR}"
