import math
from typing import Iterable

import torch


def SiLU(x: torch.Tensor) -> torch.Tensor:
    """
    component of SwiGLU
    Args:
        x:

    Returns:

    """
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim):
    """
    To realize numeric stability, use max value subtraction
    Args:
        x:
        dim:

    Returns:

    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_sub = x - x_max
    x_exp = torch.exp(x_sub)
    base = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / base


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """

    Args:
        q: (..., seq_len_q, d_head)
        k: (..., seq_len_k, d_head)
        v: (..., seq_len_k, d_head)
        mask: (seq_len_q, seq_len_k)
    Return:

    """
    d_head = q.shape[-1]
    scores = torch.einsum("...qd,...kd ->... qk", q, k) / math.sqrt(d_head)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    scores_softmax = softmax(scores, dim=-1)

    attention_weight = torch.einsum("...qk,...kd ->...qd", scores_softmax, v)
    return attention_weight

def cross_entropy(x:torch.Tensor, y:torch.Tensor):

    max_val = x.max(dim = -1, keepdim = True).values
    sum_exp = torch.exp(x - max_val).sum(dim = -1, keepdim= True)
    deno = max_val + torch.log(sum_exp)

    y_logits = x.gather(dim = -1,index = y.unsqueeze(dim = -1) )
    loss = deno - y_logits
    return loss.mean()

def learning_rate_schedule(t, lr_max, lr_min,t_w,t_c):
    """
        Given the parameters of a cosine learning rate decay schedule (with linear
        warmup) and an iteration number, return the learning rate at the given
        iteration under the specified schedule.

        Args:
            t(int): Iteration number to get learning rate for.
            lr_max(float): alpha_max, the maximum learning rate for
                cosine learning rate schedule (with warmup).
            lr_min(float): alpha_min, the minimum / final learning rate for
                the cosine learning rate schedule (with warmup).
            t_w (int): T_w, the number of iterations to linearly warm-up
                the learning rate.
            t_c(int): T_c, the number of cosine annealing iterations.

        Returns:
            Learning rate at the given iteration under the specified schedule.
        """
    if t < t_w:
        return t / t_w * lr_max
    elif t <= t_c:
        return lr_min + 0.5 * ((1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) * (lr_max - lr_min))
    return lr_min

def gradient_clipping(params:Iterable[torch.nn.Parameter], g_m:float, eps:float = 1e-6 ):
    params_with_grad = [p for p in params if p.grad is not None]

    l2_norm = math.sqrt(sum(p.grad.detach().pow(2).sum() for p in params_with_grad))
    if l2_norm >= g_m:
        factor = g_m / (l2_norm + eps)
        for p in params_with_grad:
            p.grad.detach().mul_(factor)

