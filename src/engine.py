import torch

from src.utils.adversarial import feature_attack
from src.utils.losses import TASKS, has_labels, multitask_loss
from src.utils.metrics import compute_metrics


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, device, adv_epsilon=0.05, adv_weight=0.3, grad_clip=1.0):
    """
    One epoch of adversarial training: loss = (1 - w) * clean + w * FGSM-on-features.
    The perturbation is added to the live features, so the adversarial term also trains
    the encoders, not just the fusion layers.
    """
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        if not has_labels(batch):
            continue
        out = model(batch)
        loss = multitask_loss(out, batch)
        if adv_weight > 0 and adv_epsilon > 0:
            delta = feature_attack(model, out["features"], batch, adv_epsilon)
            adv_features = {k: v + delta[k] for k, v in out["features"].items()}
            adv_loss = multitask_loss(model.classify(adv_features, batch["modality_mask"]), batch)
            loss = (1 - adv_weight) * loss + adv_weight * adv_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        n = batch["phishing"].size(0)
        total += loss.item() * n
        count += n
    return total / max(count, 1)


def _collect(results, batch, out):
    results["ids"] += batch["ids"]
    results["texts"] += batch["texts"]
    for task in TASKS:
        results["probs"][task] += out[f"logits_{task}"].softmax(dim=-1)[:, 1].detach().cpu().tolist()
        results["labels"][task] += batch[task].cpu().tolist()


def _empty_results():
    return {"ids": [], "texts": [], "probs": {t: [] for t in TASKS}, "labels": {t: [] for t in TASKS}}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    results = _empty_results()
    loss_sum, loss_n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        if has_labels(batch):
            n = batch["phishing"].size(0)
            loss_sum += multitask_loss(out, batch).item() * n
            loss_n += n
        _collect(results, batch, out)
    results["loss"] = loss_sum / loss_n if loss_n else None
    return results


def evaluate(model, loader, device, threshold=0.5):
    results = predict(model, loader, device)
    metrics = compute_metrics(results["labels"], results["probs"], threshold)
    metrics["loss"] = results["loss"]
    return metrics
