import math
import torch
from torch.optim import Optimizer
class AdamW(Optimizer):
    def __init__(self, params, lr = 1e-4, betas = (0.9,0.95), eps = 1e-8, weight_decay = 0.01):
        defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay}
        super().__init__(params,defaults)

    def step(self, closure = None):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]
                if len(state) == 0:
                    state['t'] = 0
                    state['first_moment'] = torch.zeros_like(param)
                    state['second_moment'] = torch.zeros_like(param)

                first_moment = state['first_moment']
                second_moment = state['second_moment']
                state['t'] += 1
                t = state['t']

                #m = beta1 * m + (1 - beta1) * grad
                first_moment.mul_(beta1).add_(grad, alpha = 1 - beta1)
                #v = beta2 * v + (1 - beta2) * (grad * grad)
                grad2 = grad * grad
                second_moment.mul_(beta2).add_(grad2, alpha = 1 - beta2)
                # lt = l * sqrt(1 - beta2^t) / 1 - beta1 ^t
                lt = lr * math.sqrt(1 - math.pow(beta2,t)) / (1 - math.pow(beta1,t))
                # theta = theta - lt * m / (v^ 1/2 + eps)
                param.data.addcdiv_(first_moment, torch.sqrt(second_moment) + eps, value = -lt)
                # theta = theta - lt * theta * weight_decay
                param.data.add_(param.data,alpha = -lr * weight_decay)







