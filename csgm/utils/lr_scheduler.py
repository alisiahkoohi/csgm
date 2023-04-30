# Author: Ali Siahkoohi, alisk@rice.edu
# Date: September 2022

import torch
from typing import Optional


class CustomLRScheduler(object):
    """A custom learning rate scheduler.
    The learning rate is computed as `a * (b + t) ** gamma`, where `t` is the
    iteration number, `gamma` is the decay rate, and  `a, b` are chosen to
    control the initial and final learning rate.
    Attributes:
        optimizer: A torch.optim.Optimizer to update its learning rate.
        initial_lr: A float for the initial learning rate.
        final_lr: A float for the final learning rate.
        gamma: A negative float indicating the decay rate.
        a: A float for `a` according to the initial and final learning rate
        b: A float for `b` according to the initial and final learning rate
        count: An integer for the number of steps.
    """

    def __init__(self,
                 optim: torch.optim.Optimizer,
                 initial_lr: float,
                 final_lr: float,
                 max_step: int,
                 gamma: Optional[float] = -1 / 3):
        """A custom learning rate scheduler.
        The learning rate is computed as `a * (b + k) ** gamma`, where `k` is
        the step number, `gamma` is the decay rate, and  `a,
        b` are chosen to control the initial and final learning rate.
        Args:
            optimizer: A torch.optim.Optimizer to update its learning rate.
            initial_lr: A float for the initial learning rate.
            final_lr: A float for the final learning rate.
            max_step: An integer for the maximum number of steps.
            gamma: An optional negative float indicating the decay rate.
        Raises:
            ValueError: If `final_lr` is larger than `initial_lr`.
            ValueError: If `gamma` is larger than 0.0.
        """
        super().__init__()
        if final_lr > initial_lr:
            raise ValueError('The final learning rate must be smaller than the'
                             ' initial learning rate.')
        if gamma > 0.0:
            raise ValueError('The decay rate must be negative.')

        self.optim = optim
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.gamma = gamma

        if self.initial_lr != self.final_lr:
            # Compute the a and b values for according to `initial_lr`, `final_lr`.
            self.b = max_step / ((final_lr / initial_lr)**(1 / gamma) - 1.0)
            self.a = initial_lr / (self.b**gamma)

        # Initialize the step count.
        self.count = 0

    def compute_lr(self) -> float:
        """Computes the learning rate for the current step.
        Returns:
            A float for the learning rate.
        """
        if self.initial_lr == self.final_lr:
            return self.initial_lr
        else:
            return self.a * (self.b + self.count)**self.gamma

    def step(self):
        """Updates the optimizer learning rate.
        """
        # Obtain the current learning rate.
        lr = self.compute_lr()
        # Update the optimizer learning rate.
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        # Increment the step count.
        self.count += 1
