#!/bin/zsh
# Phase 26: (1) scan Atari papers (DreamerV3 2301.04104 already has an arxiv: record — reused); (2) re-extract the 27 D4RL
# papers with 2500-word sections into phase26/vocab_records/ for the vocabulary-overlap test.
cd "$(dirname "$0")/../.."
LOG=outcome_predictor/phase26/scanner26.log; : > $LOG
for id in $(grep -v '^#' outcome_predictor/phase26/atari_papers.txt | awk '{print $2}'); do
  [ "$id" = "2301.04104" ] && continue
  grep -l "test:$id " results/*.json >/dev/null 2>&1 && { echo "SKIP $id" >> $LOG; continue; }
  venv/bin/python fulltext.py --test $id >> $LOG 2>&1; sleep 4
done
for id in $(grep -v '^#' outcome_predictor/phase26/atari_papers.txt | awk '{print $2}'); do
  f=$(grep -l "test:$id " results/*.json 2>/dev/null | head -1); [ -z "$f" ] && continue
  venv/bin/python semantic_type.py --pass v2 "$f" >> $LOG 2>&1
done
echo ATARI_DONE >> $LOG
# --- vocabulary test: long sections. New records land in results/ then are moved aside so Phase 24 records stay unique.
mkdir -p outcome_predictor/phase26/vocab_records; touch outcome_predictor/phase26/.marker
typeset -A PIN
PIN[2211.15657]='G ENERATIVE M ODELING WITH THE D ECISION D IFFUSER'
PIN[2112.10751]='R EINFORCEMENT L EARNING VIA S UPERVISED L EARNING'
PIN[2103.08050]='^4\. Fisher-BRC'
PIN[2304.12824]='^5\. Q-Guided Policy Optimization'
PIN[2401.02644]='^H IERARCHICAL D IFFUSER$'
for id in $(grep -v '^#' outcome_predictor/d4rl_papers.txt | awk '{print $2}'); do
  [ "$id" = "2006.09359" ] && continue
  if [ -n "${PIN[$id]}" ]; then venv/bin/python fulltext.py --test --words 2500 --from "${PIN[$id]}" $id >> $LOG 2>&1
  else venv/bin/python fulltext.py --test --words 2500 $id >> $LOG 2>&1; fi
  for f in $(find results -name "*.json" -newer outcome_predictor/phase26/.marker); do
    venv/bin/python semantic_type.py --pass v2 "$f" >> $LOG 2>&1; mv "$f" outcome_predictor/phase26/vocab_records/; done
  sleep 4
done
echo VOCAB_DONE >> $LOG
