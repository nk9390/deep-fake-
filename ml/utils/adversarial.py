"""
L-infinity gradient attacks (FGSM / PGD) used both for adversarial training and for measuring
how easily an attacker can push a malicious sample past the detector.
"""
import torch

from ml.utils.losses import multitask_loss


def pgd(loss_fn, inputs: dict, epsilon, steps=1, step_size=None, random_start=False):
    """
    Maximise `loss_fn(perturbed_inputs)` within an L-inf ball of radius `epsilon` around each
    tensor in `inputs`; returns the perturbations. steps=1 without random_start is FGSM.
    """
    step_size = step_size if step_size is not None else 2.5 * epsilon / steps
    originals = {k: v.detach() for k, v in inputs.items()}
    adv = {k: v.clone() for k, v in originals.items()}
    if random_start:
        adv = {k: v + torch.empty_like(v).uniform_(-epsilon, epsilon) for k, v in adv.items()}
    for _ in range(steps):
        leaves = {k: v.detach().requires_grad_(True) for k, v in adv.items()}
        loss = loss_fn(leaves)
        if not loss.requires_grad:
            break
        grads = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)
        for (k, v), g in zip(leaves.items(), grads):
            stepped = v.detach() if g is None else v.detach() + step_size * g.sign()
            adv[k] = originals[k] + (stepped - originals[k]).clamp(-epsilon, epsilon)
    return {k: adv[k] - originals[k] for k in originals}


def feature_attack(model, features, batch, epsilon, steps=1, random_start=False):
    """Perturb the per-modality embeddings before fusion (a strong, white-box upper bound)."""
    mask = batch["modality_mask"]
    return pgd(
        lambda f: multitask_loss(model.classify(f, mask), batch),
        features,
        epsilon,
        steps=steps,
        random_start=random_start,
    )


INPUT_KEYS = {"vision": "pixel_values", "audio": "audio_values"}


def input_attack(model, batch, modalities, epsilon, steps=1, random_start=False):
    """
    Perturb raw inputs (normalised pixels and/or audio samples), the realistic attacker
    setting: e.g. adding imperceptible noise to a deepfake frame so it is scored as authentic.
    """
    inputs = {INPUT_KEYS[m]: batch[INPUT_KEYS[m]] for m in modalities}
    delta = pgd(
        lambda x: multitask_loss(model({**batch, **x}), batch),
        inputs,
        epsilon,
        steps=steps,
        random_start=random_start,
    )
    return {k: batch[k] + d for k, d in delta.items()}
