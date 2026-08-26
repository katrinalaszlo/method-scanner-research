#!/bin/zsh
cd "$(dirname "$0")/../.."
LOG=outcome_predictor/phase26/scanner26.log
until grep -q "^VOCAB_DONE" $LOG; do sleep 20; done
venv/bin/python outcome_predictor/phase26/vocab_test.py > outcome_predictor/phase26/vocab_test.log 2>&1 &
mkdir -p outcome_predictor/phase26/superseded_records
typeset -A PIN
PIN[1903.00374]='^W ORLD M ODELS$'
PIN[1906.05243]='^4\. Model-based algorithms at scale'
for id in ${(k)PIN}; do
  for f in $(grep -l "test:$id " results/*.json); do mv "$f" outcome_predictor/phase26/superseded_records/; echo "MOVED $f" >> $LOG; done
  venv/bin/python fulltext.py --test --from "${PIN[$id]}" $id >> $LOG 2>&1; sleep 4
  f=$(grep -l "test:$id " results/*.json | head -1); venv/bin/python semantic_type.py --pass v2 "$f" >> $LOG 2>&1
done
echo PINS_DONE >> $LOG
wait
echo ALL_DONE >> $LOG
