#!/bin/bash
# setup_remotes.sh — Configure git remotes for any server
#
# Run ONCE on each server/machine to set up the two-repo architecture.
#
# Usage: bash scripts/setup_remotes.sh
#
# After running this, you can use:
#   git push private master     # push everything (papers included)
#   git push public master      # push without papers
#   bash scripts/sync_to_public.sh  # push to BOTH in one step

set -e

echo "Setting up PWM git remotes..."

# Check current remotes
CURRENT=$(git remote)

# Handle existing 'origin' remote
if echo "$CURRENT" | grep -q '^origin$'; then
    URL=$(git remote get-url origin)
    if echo "$URL" | grep -q 'Physics_World_Model'; then
        echo "  Renaming origin -> public (Physics_World_Model)"
        git remote rename origin public
    elif echo "$URL" | grep -q 'pwm'; then
        echo "  Renaming origin -> private (pwm)"
        git remote rename origin private
    fi
fi

# Add private remote if missing
if ! git remote get-url private > /dev/null 2>&1; then
    echo "  Adding remote: private -> pwm"
    git remote add private https://github.com/integritynoble/pwm.git
else
    echo "  Remote 'private' already exists"
fi

# Add public remote if missing
if ! git remote get-url public > /dev/null 2>&1; then
    echo "  Adding remote: public -> Physics_World_Model"
    git remote add public https://github.com/integritynoble/Physics_World_Model.git
else
    echo "  Remote 'public' already exists"
fi

# Set default push to private (so 'git push' goes to private)
git config push.default current
git branch --set-upstream-to=private/master master 2>/dev/null || true

echo ""
echo "Remotes configured:"
git remote -v
echo ""
echo "Default push: private (pwm)"
echo "To sync both: bash scripts/sync_to_public.sh"
