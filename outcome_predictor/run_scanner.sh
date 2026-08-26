#!/bin/zsh
# Phase 24: extract + v2-type every D4RL paper not already in results/. Diffuser (2205.09991) already has a typed record.
cd "$(dirname "$0")/.."
LOG=outcome_predictor/scanner_run.log
: > $LOG
for id in $(grep -v '^#' outcome_predictor/d4rl_papers.txt | awk '{print $2}'); do
  [ "$id" = "2205.09991" ] && continue
  if grep -l "test:$id " results/*.json >/dev/null 2>&1; then echo "SKIP $id (record exists)" >> $LOG; continue; fi
  venv/bin/python fulltext.py --test $id >> $LOG 2>&1
  sleep 4
done
for id in $(grep -v '^#' outcome_predictor/d4rl_papers.txt | awk '{print $2}'); do
  f=$(grep -l "test:$id " results/*.json 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "NOREC $id" >> $LOG; continue; }
  venv/bin/python semantic_type.py --pass v2 "$f" >> $LOG 2>&1
done
echo DONE >> $LOG
