#!/bin/bash
# sync_to_public.sh — Push to public repo (Physics_World_Model) excluding papers/
#
# Usage: bash scripts/sync_to_public.sh
#
# Remotes (set up once per machine):
#   private → https://github.com/integritynoble/pwm.git       (full project)
#   public  → https://github.com/integritynoble/Physics_World_Model (no papers)
#
# How it works:
#   1. Pushes everything to private repo (papers included)
#   2. Creates a temporary branch with papers/ removed from index
#   3. Force-pushes that filtered branch to public repo
#   4. Cleans up, returns to master
#
# Multi-server setup:
#   Run this on any server after committing. All servers should have
#   both remotes configured (see setup_remotes.sh).

set -e

echo "============================================"
echo "  Syncing: private (full) + public (no papers)"
echo "============================================"

# Sanity checks
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "master" ]; then
    echo "ERROR: Not on master branch (on $BRANCH). Switch to master first."
    exit 1
fi

# Check remotes exist
git remote get-url private > /dev/null 2>&1 || { echo "ERROR: 'private' remote not set. Run setup_remotes.sh first."; exit 1; }
git remote get-url public  > /dev/null 2>&1 || { echo "ERROR: 'public' remote not set. Run setup_remotes.sh first."; exit 1; }

# Stash any uncommitted changes
STASHED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Stashing uncommitted changes..."
    git stash push -m "sync_to_public auto-stash"
    STASHED=true
fi

# 1. Push everything to private
echo ""
echo "[1/4] Pushing to private (full content)..."
git push private master

# 2. Create filtered branch (papers removed from index, NOT from disk)
echo ""
echo "[2/4] Creating filtered snapshot (no papers/)..."
git checkout -b _public_sync master
git rm -r --cached papers/ > /dev/null 2>&1 || true

# Add papers/ to .gitignore if not already there
if ! grep -q '^papers/' .gitignore 2>/dev/null; then
    echo "papers/" >> .gitignore
    git add .gitignore
fi

git commit -m "sync: public snapshot (papers excluded)" --allow-empty > /dev/null

# 3. Push to public
echo ""
echo "[3/4] Pushing to public..."
git push public _public_sync:master --force

# 4. Clean up
echo ""
echo "[4/4] Cleaning up..."
git checkout master
git branch -D _public_sync > /dev/null

# Restore stashed changes
if [ "$STASHED" = true ]; then
    echo "Restoring stashed changes..."
    git stash pop
fi

echo ""
echo "============================================"
echo "  Done!"
echo "  private (pwm):              full content"
echo "  public (Physics_World_Model): no papers/"
echo "============================================"
