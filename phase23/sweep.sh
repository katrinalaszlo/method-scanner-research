#!/bin/bash
# Phase 23 sweep: seeds outer so a partial run still covers every config for seed 0.
cd "/Users/katlaszlo/Desktop/Desktop - Kat’s MacBook Air/Github-Wiki/GitHub/method_scanner/phase23/diffuser"
export PYTHONPATH=../shims:. DIFFUSER_DEVICE=mps
OUT=../results.csv
[ -f $OUT ] || echo "env,sampler,steps,seed,return,score,ep_len,terminal,seconds" > $OUT
for seed in ${SEEDS:-0 1 2}; do
  for env in halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2; do
    for cfg in "ddpm 20" "ddim 20" "ddim 10" "ddim 5" "ddim 2"; do
      set -- $cfg; sampler=$1; steps=$2
      suffix="p23_${sampler}${steps}_s${seed}"
      if grep -q "^${env},${sampler},${steps},${seed}," $OUT; then continue; fi
      t0=$(date +%s)
      ../venv/bin/python scripts/plan_guided.py --dataset $env --loadbase logs/pretrained --seed $seed --suffix $suffix --vis_freq 100000 \
        --sampler $sampler --ddim_steps $steps > ../runs_${suffix}_${env}.log 2>&1
      t1=$(date +%s)
      j=logs/$env/plans/defaults/$suffix/rollout.json
      if [ -f "$j" ]; then
        ../venv/bin/python -c "import json,sys; d=json.load(open('$j')); print(f\"$env,$sampler,$steps,$seed,{d['return']:.2f},{d['score']:.4f},{d['step']+1},{int(d['term'])},$((t1-t0))\")" >> $OUT
      else
        echo "$env,$sampler,$steps,$seed,NA,NA,NA,NA,$((t1-t0))" >> $OUT
      fi
      tail -1 $OUT
    done
  done
done
echo "SWEEP DONE"
