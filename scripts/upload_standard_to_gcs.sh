#!/bin/bash
# Upload all standard datasets to GCS
# Run: bash scripts/upload_standard_to_gcs.sh
#
# Prerequisites:
#   gcloud auth login
#
# GCS path: gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/
# Total size: ~1 GB across 170 modalities

set -e

GCS_BUCKET="gs://pwm-benchmark-datasets/datasets/Benchmark"
LOCAL_BASE="D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark"

echo "Checking GCS auth..."
gcloud auth print-access-token > /dev/null 2>&1 || {
    echo "ERROR: Not authenticated. Run: gcloud auth login"
    exit 1
}

echo "Uploading standard datasets to $GCS_BUCKET/"
echo "============================================================"

count=0
total=0
for mod_dir in "$LOCAL_BASE"/*/standard/; do
    if [ ! -d "$mod_dir" ]; then continue; fi
    mod_name=$(basename $(dirname "$mod_dir"))
    if [[ "$mod_name" == _* ]]; then continue; fi

    total=$((total + 1))
    n_h5=$(ls "$mod_dir"*.h5 2>/dev/null | wc -l)
    echo -n "[$total] $mod_name ($n_h5 h5 files)... "

    # Upload all h5, json, and image files
    gcloud storage cp "$mod_dir"*.h5 "$GCS_BUCKET/$mod_name/standard/" --quiet 2>/dev/null
    gcloud storage cp "$mod_dir"*.json "$GCS_BUCKET/$mod_name/standard/" --quiet 2>/dev/null

    if [ -d "$mod_dir/images" ]; then
        gcloud storage cp "$mod_dir/images/"*.png "$GCS_BUCKET/$mod_name/standard/images/" --quiet 2>/dev/null
    fi

    echo "OK"
    count=$((count + 1))
done

echo ""
echo "============================================================"
echo "Upload complete: $count/$total modalities"
echo "GCS path: $GCS_BUCKET/{modality}/standard/"
