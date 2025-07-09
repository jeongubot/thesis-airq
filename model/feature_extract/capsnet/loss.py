# model/capsnet/loss.py
import torch
import torch.nn.functional as F

def margin_loss(y_true, y_pred_lengths, m_pos=0.9, m_neg=0.1, lam=0.5):
    """
    y_true: one-hot [batch, num_classes]
    y_pred_lengths: lengths of capsule outputs [batch, num_classes]
    """
    zero = torch.zeros_like(y_pred_lengths)
    # max(0, m_pos - v_c)^2 for true class
    loss_pos = torch.max(zero, m_pos - y_pred_lengths) ** 2
    # max(0, v_c - m_neg)^2 for other classes
    loss_neg = torch.max(zero, y_pred_lengths - m_neg) ** 2

    L = y_true * loss_pos + lam * (1 - y_true) * loss_neg
    return L.sum(dim=1).mean()
