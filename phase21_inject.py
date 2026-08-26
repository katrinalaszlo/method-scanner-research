"""Phase 21: delta-paper test. Prepend the base paper's Algorithm 1 to a delta paper's pinned method section, then extract.
Usage: venv/bin/python phase21_inject.py     (writes sections to predictions/phase21_sections/, posts to /api/extract)"""
import os
from fulltext import arxiv_meta, fetch_text, pinned_section, post_extract, ROOT

DELTA_WORDS = 1200
OUT = os.path.join(ROOT, "predictions", "phase21_sections")

def base_block(arxiv_id, start, end):
    lines = fetch_text(arxiv_id).split("\n")[start - 1:end]
    return "\n".join(l for l in lines if l.strip() not in ("", "0"))  # drop pdftotext's stray superscript lines

BASES = {
    "DDPM": ("2006.11239", 274, 305, "Denoising Diffusion Probabilistic Models (Ho et al. 2020)"),
    "MAML": ("1703.03400", 160, 172, "Model-Agnostic Meta-Learning (Finn et al. 2017)"),
    "DQN": ("1312.5602", 254, 276, "Playing Atari with Deep Reinforcement Learning (Mnih et al. 2013)"),
    "DDPG": ("1509.02971", 241, 275, "Continuous control with deep reinforcement learning (Lillicrap et al. 2015)"),
}
DELTAS = [
    ("1509.06461", "DQN", r"^Double DQN$"),
    ("1802.09477", "DDPG", r"^4\.2\. Clipped Double Q-Learning"),
    ("1710.02298", "DQN", r"^The Integrated Agent$"),
    ("2010.02502", "DDPM", r"ARIATIONAL I ?NFERENCE FOR N ?ON"),   # Phase 22: same section the corpus record used
    ("1803.02999", "MAML", r"^Meta-Learning an Initialization$"),
]
import sys
if len(sys.argv) > 1:
    DELTAS = [d for d in DELTAS if d[0] in sys.argv[1:]]

os.makedirs(OUT, exist_ok=True)
for aid, base, pin in DELTAS:
    bid, s, e, bname = BASES[base]
    title, _ = arxiv_meta(aid)
    delta, how = pinned_section(fetch_text(aid), pin, DELTA_WORDS)
    text = (f"This method builds on the following base algorithm from {bname}, which it inherits unchanged except where noted below.\n\n"
            f"{base_block(bid, s, e)}\n\nThe modification introduced in this paper:\n\n{delta}")
    open(os.path.join(OUT, f"{aid}.txt"), "w").write(text)
    res = post_extract(title, text, f"arxiv:{aid} [{'diffusion' if base == 'DDPM' else 'meta-learning' if base == 'MAML' else 'policy-learning'}] injected: {base} Algorithm 1 + {how}")
    d = res.get("result", {})
    print(f"{'OK  ' if res.get('success') else 'FAIL'} {aid} {title[:40]:42} {len(text.split())}w ops={d.get('data', {}).get('operations')} canon={d.get('canonical_ops')} {res.get('error') or ''}", flush=True)
