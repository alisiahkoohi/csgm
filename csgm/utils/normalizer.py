import torch
from typing import Optional


class Normalizer():
    """Normalizer a tensor image with training mean and standard deviation.

    Extracts the mean and standard deviation from the training dataset, and uses
    them to normalize an input image.

    Attributes:
        mean: A torch.Tensor containing the mean over the training dataset.
        std: A torch.Tensor containing the standard deviation over the training.
        eps: A small float to avoid dividing by 0.
    """

    def __init__(self, dataset: torch.Tensor, eps: Optional[int] = 0.00001):
        """Initializes a Normalizer object.

        Args:
            dataset: A torch.Tensor that first dimension is the batch dimension.
            eps: A optional small float to avoid dividing by 0.
        """
        super().__init__()

        # Compute the training dataset mean and standard deviation over the
        # batch dimensions.
        self.mean = torch.mean(dataset, 0)
        self.std = torch.std(dataset, 0)
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input sample.

        Args:
            x: A torch.Tensor with the same dimension organization as `dataset`.

        Returns:
            A torch.Tensor with the same dimension organization as `x` but
            normalized with the mean and standard deviation of the training
            dataset.
        """
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Restore the normalization from the input sample.

        Args:
            x: A normalized torch.Tensor with the same dimension organization as
            `dataset`.

        Returns:
            A torch.Tensor with the same dimension organization as `x` that has
            been unnormalized.
        """
        return x * (self.std + self.eps) + self.mean
