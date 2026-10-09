import torch.nn.functional as F

TASKS = ("phishing", "deepfake")


def has_labels(batch) -> bool:
    return any(bool((batch[task] >= 0).any()) for task in TASKS)


def multitask_loss(out, batch):
    """Sum of per-task cross-entropy over labelled samples (label -1 = unlabelled, ignored)."""
    loss = out["logits_phishing"].new_zeros(())
    for task in TASKS:
        labels = batch[task]
        valid = labels >= 0
        if valid.any():
            loss = loss + F.cross_entropy(out[f"logits_{task}"][valid], labels[valid])
    return loss
