import copy

import torch
import torch.nn as nn
from mmcv.runner import OptimizerHook,HOOKS

@HOOKS.register_module()
class MetaOptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
    """

    def after_train_iter(self, runner):
        runner.optimizer['classifier'].zero_grad()
        runner.outputs['loss'].backward()
        
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer['classifier'].step()