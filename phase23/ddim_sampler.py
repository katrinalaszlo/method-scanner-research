"""Phase 23: DDIM sampler for Diffuser (Song et al. 2020, eta=0 -> deterministic).

Drop-in alternative to GaussianDiffusion.p_sample_loop. Uses the trained model's own
alphas_cumprod on a strided subset of its T timesteps, so a T=20 checkpoint can be sampled
with 20, 10, 5 or 2 steps. Value guidance is applied exactly as in n_step_guided_p_sample
(same scale, t_stopgrad, n_guide_steps, scale_grad_by_std) so only the reverse step changes.
"""
import torch

from diffuser.models.helpers import extract, apply_conditioning
from diffuser.models.diffusion import Sample, sort_by_values, make_timesteps
import diffuser.utils as utils


def ddim_timesteps(n_train_steps, n_sample_steps):
    """Strided subset of [0, T), descending, always ending at 0. e.g. T=20, 5 -> [19, 15, 11, 7, 3]."""
    assert 1 <= n_sample_steps <= n_train_steps
    stride = n_train_steps / n_sample_steps
    ts = sorted({int(round(i * stride)) for i in range(n_sample_steps)}, reverse=True)
    return ts


@torch.no_grad()
def ddim_guided_sample_loop(
    model, shape, cond, guide=None, ddim_steps=None, ddim_eta=0.0,
    scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
    verbose=True, return_chain=False, **unused,
):
    device = model.betas.device
    batch_size = shape[0]
    ddim_steps = ddim_steps or model.n_timesteps
    ts = ddim_timesteps(model.n_timesteps, ddim_steps)
    ts_prev = ts[1:] + [-1]  # -1 means "final step: x0 with alpha_prev = 1"

    x = torch.randn(shape, device=device)
    x = apply_conditioning(x, cond, model.action_dim)
    chain = [x] if return_chain else None
    values = torch.zeros(batch_size, device=device)

    progress = utils.Progress(len(ts)) if verbose else utils.Silent()
    for i, i_prev in zip(ts, ts_prev):
        t = make_timesteps(batch_size, i, device)

        ## value guidance — identical to sampling/functions.py
        if guide is not None:
            model_var = torch.exp(extract(model.posterior_log_variance_clipped, t, x.shape))
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    values, grad = guide.gradients(x, cond, t)
                if scale_grad_by_std:
                    grad = model_var * grad
                grad[t < t_stopgrad] = 0
                x = x + scale * grad
                x = apply_conditioning(x, cond, model.action_dim)

        ## DDIM reverse step
        eps = model.model(x, cond, t)
        x0 = model.predict_start_from_noise(x, t=t, noise=eps)
        if model.clip_denoised:
            x0.clamp_(-1., 1.)
        if model.predict_epsilon is False:
            # model predicts x0 directly; recover eps for the direction term
            a_t = extract(model.alphas_cumprod, t, x.shape)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()

        a_t = extract(model.alphas_cumprod, t, x.shape)
        if i_prev >= 0:
            a_prev = extract(model.alphas_cumprod, make_timesteps(batch_size, i_prev, device), x.shape)
        else:
            a_prev = torch.ones_like(a_t)
        sigma = ddim_eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
        direction = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        noise = torch.randn_like(x) if ddim_eta > 0 and i_prev >= 0 else torch.zeros_like(x)
        x = a_prev.sqrt() * x0 + direction + sigma * noise
        x = apply_conditioning(x, cond, model.action_dim)

        progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
        if return_chain: chain.append(x)

    progress.stamp()
    x, values = sort_by_values(x, values)
    if return_chain: chain = torch.stack(chain, dim=1)
    return Sample(x, values, chain)
