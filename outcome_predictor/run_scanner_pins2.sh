#!/bin/zsh
# Phase 24: Hierarchical Diffuser auto-section was "Flat Learning Methods" (background). Pin to section 3.
cd "$(dirname "$0")/.."
LOG=outcome_predictor/scanner_run.log
until grep -q "^DONE2" $LOG; do sleep 15; done
id=2401.02644
for f in $(grep -l "test:$id " results/*.json); do mv "$f" outcome_predictor/superseded_records/; echo "MOVED $f" >> $LOG; done
venv/bin/python fulltext.py --test --from '^H IERARCHICAL D IFFUSER$' $id >> $LOG 2>&1
f=$(grep -l "test:$id " results/*.json | head -1)
venv/bin/python semantic_type.py --pass v2 "$f" >> $LOG 2>&1
echo DONE3 >> $LOG
