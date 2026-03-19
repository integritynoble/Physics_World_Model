#!/bin/bash
set -e

echo "=== Syncing Physics_World_Model (public) ==="
cd ~/workspace/Physics_World_Model
git fetch origin
git pull origin master

echo "=== Syncing pwm (private research) ==="
cd ~/workspace/pwm
git fetch origin
git pull origin main
git submodule update --init --recursive

echo "=== Syncing pwm_product (private product) ==="
cd ~/workspace/pwm_product
git fetch origin
git pull origin main
git submodule update --init --recursive

echo "=== All repos synced ==="
