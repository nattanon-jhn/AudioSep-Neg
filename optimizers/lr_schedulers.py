from functools import partial
from typing import Callable


# ============================================================
# Linear warm-up + step decay
# ============================================================
def linear_warm_up(
    step: int,
    warm_up_steps: int,
    reduce_lr_steps: int
) -> float:
    """
    Linear warm-up followed by step decay (0.9^k).

    Args:
        step (int): global step from Lightning
        warm_up_steps (int): number of warm-up steps
        reduce_lr_steps (int): decay interval (steps)

    Returns:
        lr_scale (float): learning rate multiplier
    """

    # safety: no warm-up
    if warm_up_steps <= 0:
        return 1.0

    # warm-up phase
    if step < warm_up_steps:
        return step / float(warm_up_steps)

    # decay phase
    reduce_lr_steps = max(reduce_lr_steps, 1)
    return 0.9 ** (step // reduce_lr_steps)


# ============================================================
# Piecewise constant warm-up (legacy / optional)
# ============================================================
def constant_warm_up(
    step: int,
    warm_up_steps: int,
    reduce_lr_steps: int
) -> float:
    """
    Piecewise constant warm-up.
    Mostly kept for backward compatibility.

    Returns:
        lr_scale (float)
    """

    if warm_up_steps <= 0:
        return 1.0

    if 0 <= step < warm_up_steps:
        return 0.001
    elif warm_up_steps <= step < 2 * warm_up_steps:
        return 0.01
    elif 2 * warm_up_steps <= step < 3 * warm_up_steps:
        return 0.1
    else:
        return 1.0


# ============================================================
# Factory
# ============================================================
def get_lr_lambda(
    lr_lambda_type: str,
    **kwargs
) -> Callable:
    """
    Get LR lambda function for torch.optim.lr_scheduler.LambdaLR

    Args:
        lr_lambda_type (str):
            - "linear_warm_up"
            - "constant_warm_up"

    Returns:
        lr_lambda_func (Callable)
    """

    if lr_lambda_type == "linear_warm_up":
        return partial(
            linear_warm_up,
            warm_up_steps=kwargs.get("warm_up_steps", 0),
            reduce_lr_steps=kwargs.get("reduce_lr_steps", 1),
        )

    elif lr_lambda_type == "constant_warm_up":
        return partial(
            constant_warm_up,
            warm_up_steps=kwargs.get("warm_up_steps", 0),
            reduce_lr_steps=kwargs.get("reduce_lr_steps", 1),
        )

    else:
        raise NotImplementedError(
            f"Unknown lr_lambda_type: {lr_lambda_type}"
        )
