#!/bin/bash
# Phase 24: eta sweep. DDIM at 10 and 20 steps, eta in {0.5, 1.0}, hopper + walker, seeds 0-2.
cd "/Users/katlaszlo/Desktop/Desktop - Kat’s MacBook Air/Github-Wiki/GitHub/method_scanner/phase23/diffuser"
export PYTHONPATH=../shims:. DIFFUSER_DEVICE=mps
for seed in 0 1 2; do
  for env in hopper-medium-v2 walker2d-medium-v2; do
    for steps in 10 20; do
      for eta in 0.5 1.0; do
        suffix="p24_ddim${steps}_eta${eta}_s${seed}"
        if ls logs/$env/plans/*/$suffix/rollout.json >/dev/null 2>&1; then continue; fi
        t0=$(date +%s)
        ../venv/bin/python scripts/plan_guided.py --dataset $env --loadbase logs/pretrained --seed $seed --suffix $suffix --vis_freq 100000 \
          --sampler ddim --ddim_steps $steps --ddim_eta $eta > ../runs_${suffix}_${env}.log 2>&1
        echo "$env,ddim,$steps,$eta,$seed,$(( $(date +%s) - t0 ))s"
      done
    done
  done
done
echo "ETA SWEEP DONE"
