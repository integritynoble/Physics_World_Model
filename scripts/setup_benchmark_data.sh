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

# Available benchmark datasets
DATASETS=(sd_cassi cacti spc_kronecker)

usage() {
    echo "Usage: $0 [OPTIONS] [DATASET...]"
    echo ""
    echo "Downloads benchmark datasets from GCS."
    echo ""
    echo "Datasets: ${DATASETS[*]}"
    echo ""
    echo "Options:"
    echo "  --challenge    Download only challenge HDF5 files (no source images)"
    echo "  --list         List available datasets and their sizes on GCS"
    echo "  --help         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                     # Download all datasets (source + challenge)"
    echo "  $0 sd_cassi cacti      # Download specific datasets"
    echo "  $0 --challenge         # Download only challenge HDF5 files"
    echo "  $0 --challenge cacti   # Download only cacti challenge files"
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

# Parse arguments
CHALLENGE_ONLY=false
SELECTED=()

for arg in "$@"; do
    case "$arg" in
        --challenge) CHALLENGE_ONLY=true ;;
        --list) list_datasets; exit 0 ;;
        --help|-h) usage; exit 0 ;;
        *) SELECTED+=("$arg") ;;
    esac
done

# Default: all datasets
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("${DATASETS[@]}")
fi

# Verify gsutil is available
if ! command -v gsutil &>/dev/null; then
    echo "ERROR: gsutil not found. Install Google Cloud SDK first:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "PWM Benchmark Data Setup"
echo "========================"
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

echo "All done! Benchmark data is ready at: ${BENCHMARK_DIR}"
