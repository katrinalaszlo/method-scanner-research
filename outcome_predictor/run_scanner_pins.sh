#!/bin/zsh
# Phase 24: re-extract papers whose auto-detected section was wrong (pinned to the method section), then type v2.
# Pins chosen from the section headings in papers/<id>.txt BEFORE any evaluation was run.
cd "$(dirname "$0")/.."
LOG=outcome_predictor/scanner_run.log
until grep -q "^DONE" $LOG; do sleep 15; done
mkdir -p outcome_predictor/superseded_records
typeset -A PIN
PIN[2211.15657]='G ENERATIVE M ODELING WITH THE D ECISION D IFFUSER'
PIN[2112.10751]='R EINFORCEMENT L EARNING VIA S UPERVISED L EARNING'
PIN[2103.08050]='^4\. Fisher-BRC'
PIN[2304.12824]='^5\. Q-Guided Policy Optimization'
for id in ${(k)PIN}; do
  for f in $(grep -l "test:$id " results/*.json); do mv "$f" outcome_predictor/superseded_records/; echo "MOVED $f" >> $LOG; done
  venv/bin/python fulltext.py --test --from "${PIN[$id]}" $id >> $LOG 2>&1
  sleep 4
  f=$(grep -l "test:$id " results/*.json | head -1)
  venv/bin/python semantic_type.py --pass v2 "$f" >> $LOG 2>&1
done
echo DONE2 >> $LOG
