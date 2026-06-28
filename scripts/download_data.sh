#!/usr/bin/env bash
# Download the PTCG AI Battle Challenge engine + data from Kaggle.
#
# PREREQUISITES (one-time):
#   1. Join the competition (accept rules) in your browser — you CANNOT download
#      otherwise, the API returns 403:
#        https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/rules
#   2. Create an API token: kaggle.com -> Settings -> "Create New Token".
#      This downloads kaggle.json. Then:
#        mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#
# Then run:  bash scripts/download_data.sh
set -euo pipefail

cd "$(dirname "$0")/.."
KAGGLE="${KAGGLE:-.venv/bin/kaggle}"
COMP="pokemon-tcg-ai-battle-challenge-strategy"
SIM_COMP="pokemon-tcg-ai-battle"   # sibling Simulation competition (often hosts the engine/episodes)

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found. See header of this script for setup." >&2
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json" 2>/dev/null || true

echo "==> Listing competition files for $COMP"
"$KAGGLE" competitions files -c "$COMP" || {
  echo "If this 403s, you have not accepted the competition rules in the browser yet." >&2
  exit 1
}

echo "==> Downloading $COMP into ./engine/"
mkdir -p engine
"$KAGGLE" competitions download -c "$COMP" -p engine
echo "==> Unzipping"
( cd engine && for z in *.zip; do [ -f "$z" ] && unzip -o "$z" && rm -f "$z"; done )

echo
echo "Done. Inspect ./engine/ for the simulator package (look for a 'cg/' folder,"
echo "sample main.py agents, and the obs_dict / action interface docs)."
echo
echo "If the engine lives on the sibling Simulation competition instead, run:"
echo "  $KAGGLE competitions download -c $SIM_COMP -p engine"
