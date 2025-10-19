import pytest
import torch
from orix.quaternion import symmetry as SYM
from training.loss_functions import (
    build_loss,
    fz_reduced_rotational_distance_loss,
    rotational_distance_loss,
)


# Define a test case
def test_loss_function_with_symmetry():
    # Example configuration for the loss and symmetry group
    cfg = {
        "loss": "fz_reduced_rotational_distance",  # 'fz_reduced_rotational_distance_loss'
        "symmetry_group": "Oh",  # 'Oh' symmetry group
    }

    # Build the loss function based on the configuration
    loss_fn = build_loss(cfg)

    # Create dummy quaternion data
    q_pred = torch.randn(5, 4, 128, 128)  # Example predicted quaternions (batch of 5)
    q_target = torch.randn(5, 4, 128, 128)  # Example target quaternions (batch of 5)

    # Run the loss function on the dummy data
    loss = loss_fn(q_pred, q_pred)

    # Check if the loss is a tensor (i.e., the function ran correctly)
    assert isinstance(
        loss, torch.Tensor
    ), "The output of the loss function should be a torch tensor."
    assert loss.item() >= 0, "The loss should be non-negative."

    # Optionally print loss for debugging purposes (can be removed in production)
    print(f"Loss: {loss.item()}")


# To run the test:
# pytest tests/test_loss_function.py --maxfail=1 --disable-warnings -q
