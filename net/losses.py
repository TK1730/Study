import torch
import torch.nn.functional as F

def kl_loss(m_v, logv_v, m_w, logv_w, frame_length=1):
    """
    m_v, v_v: mean and variance of voice
    m_w, v_w: mean and variance of whisper
    """
    m_v = m_v.float()
    logv_v = logv_v.float()
    m_w = m_w.float()
    logv_w = logv_w.float()

    # Compute variance from log variance
    var_v = torch.exp(logv_v)  # Variance of voiced
    var_w = torch.exp(logv_w)  # Variance of whisper

    # KL divergence formula
    kl = logv_w - logv_v - 0.5  # log(σ2^2 / σ1^2) - 1/2
    kl += 0.5 * ((var_v + (m_v - m_w) ** 2) / var_w)  # (σ1^2 + (μ1 - μ2)^2) / σ2^2

    # Sum over dimensions and normalize by frame length

    return torch.mean(kl)


def feature_vec_1(v, eps=1e-9):
    """
        特徴量を一次元にしてl2で正規化
    """
    v = torch.flatten(v, start_dim=1)
    v_l2 = torch.sqrt((v**2).sum(dim=1))
    v = (v.T/(v_l2+eps)).T
    return v

def featire_vec_2(v, eps=1e-9):
    """
        時間方向に平均をとってl2で正規化
        gloval average pooling
    """
    v = torch.mean(v, dim=2)
    vf_l2 = torch.sqrt((v**2).sum(dim=1))
    v = (v.T/(vf_l2+eps)).T
    return v

def contrast_loss(v_f, w_f, t=1.0, eps=1e-9, device='cpu'):
    """
    v_f, w_f: [b, c, w]
    t: temprature
    eps: epsilon

    return: contrastive loss
    """
    logits = torch.matmul(v_f, w_f.T) * torch.exp(torch.tensor(t))
    labels = torch.arange(0, v_f.size(0), dtype=torch.long, device=device)
    entropy_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return entropy_loss, logits
