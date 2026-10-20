"""Activation steering with a retain language-model objective.

The steering component follows https://github.com/centerforaisafety/wmdp.
"""

import re
import torch


def masked_activation_mse(activation, target, mask):
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Each example must contain supervised answer tokens.")
    squared_error = (activation.float() - target.float()).square().mean(dim=-1)
    return ((squared_error * mask).sum(dim=1) / counts).mean()


def select_parameters(model, patterns):
    selected = []
    for name, param in model.named_parameters():
        param.requires_grad_(any(re.fullmatch(pattern, name) for pattern in patterns))
        if param.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError("No parameters matched --trainable-regex.")
    return selected


class ConflictObjective:
    def __init__(self, model, module_regex, steering=20.0, forget_weight=1.0, retain_weight=1.0):
        modules = [
            module for name, module in model.named_modules() if re.fullmatch(module_regex, name)
        ]
        if len(modules) != 1:
            raise ValueError("The activation module regex must match exactly one module.")
        self.module = modules[0]
        self.steering = steering
        self.forget_weight, self.retain_weight = forget_weight, retain_weight
        self.control = None

    def __call__(self, model, batch):
        activations = []

        def capture(module, inputs, output):
            activations.append(output[0] if isinstance(output, tuple) else output)

        hook = self.module.register_forward_hook(capture)
        try:
            outputs = model(**batch["forget"])
        finally:
            hook.remove()
        activation = activations[0]
        if self.control is None:
            vector = torch.rand(1, 1, activation.shape[-1])
            self.control = vector / vector.norm() * self.steering
        target = self.control.to(activation.device, activation.dtype)
        forget_loss = masked_activation_mse(activation, target, batch["forget"]["labels"].ne(-100))
        retain_loss = model(**batch["retain"]).loss
        return (
            self.forget_weight * forget_loss + self.retain_weight * retain_loss,
            outputs,
        )
