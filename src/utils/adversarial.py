import torch


def fgsm_on_features(model, features, targets, epsilon=0.05):
    """
    Apply FGSM-style adversarial perturbation on modality features prior to fusion.
    features: dict with 'text_feat', 'audio_feat', 'vision_feat' (each [B, D])
    targets: dict with 'phishing' (LongTensor), 'deepfake' (LongTensor)
    """
    # Clone and prepare for grad
    adv_feats = {}
    for k in ["text_feat", "audio_feat", "vision_feat"]:
        feat = features[k].detach().requires_grad_(True)
        adv_feats[k] = feat

    out = model.forward_from_features(adv_feats)
    loss = torch.nn.functional.cross_entropy(out["logits_phishing"], targets["phishing"]) + \
           torch.nn.functional.cross_entropy(out["logits_deepfake"], targets["deepfake"])
    loss.backward()

    with torch.no_grad():
        for k in ["text_feat", "audio_feat", "vision_feat"]:
            grad = adv_feats[k].grad
            if grad is None:
                continue
            adv_feats[k] = adv_feats[k] + epsilon * torch.sign(grad)
    return adv_feats