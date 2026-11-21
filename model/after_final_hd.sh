#!/bin/bash
# Waits for model/phase1_final.py to finish (its draft marker), then renders the HD draft on model/css/final.
cd "$(dirname "$0")/.."
until [ -f model/phase1_final.draft_done ]; do sleep 60; done
OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n viz python -I render/render.py --draft --hd --css model/css/final --workers 6 \
  > render/draft_final_hd.out 2>&1
echo "done $(date)" >> render/draft_final_hd.out
