#!/bin/zsh
cd "$(dirname "$0")/../.."
LOG=outcome_predictor/phase26/scanner26.log; echo "--- vocab loop restart with num_ctx 8192" >> $LOG
touch outcome_predictor/phase26/.marker
typeset -A PIN
PIN[2211.15657]='G ENERATIVE M ODELING WITH THE D ECISION D IFFUSER'
PIN[2112.10751]='R EINFORCEMENT L EARNING VIA S UPERVISED L EARNING'
PIN[2103.08050]='^4\. Fisher-BRC'
PIN[2304.12824]='^5\. Q-Guided Policy Optimization'
PIN[2401.02644]='^H IERARCHICAL D IFFUSER$'
for id in $(grep -v '^#' outcome_predictor/d4rl_papers.txt | awk '{print $2}'); do
  [ "$id" = "2006.09359" ] && continue
  grep -l "test:$id " outcome_predictor/phase26/vocab_records/*.json >/dev/null 2>&1 && continue
  if [ -n "${PIN[$id]}" ]; then venv/bin/python fulltext.py --test --words 2500 --from "${PIN[$id]}" $id >> $LOG 2>&1
  else venv/bin/python fulltext.py --test --words 2500 $id >> $LOG 2>&1; fi
  for f in $(find results -name "*.json" -newer outcome_predictor/phase26/.marker); do
    venv/bin/python semantic_type.py --pass v2 "$f" >> $LOG 2>&1; mv "$f" outcome_predictor/phase26/vocab_records/; done
  sleep 4
done
echo VOCAB_DONE >> $LOG
