import os
import json
import datetime
import numpy as np
#import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Generate model name automatically
# -----------------------------------------------------------
def make_model_name(
    blocks=4,
    filters=64,
    input_shape="12x5000",
    loss="bce",
    aug=False,
    seed=42,
    prefix="resnet"
):
    """
    Returns a descriptive model name, example:
    resnet_6b_128f_12x5000_bce_aug_s42_20251205-2245
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    aug_flag = "aug" if aug else "noaug"

    name = (
        f"{prefix}_{blocks}b_"
        f"{input_shape}_"
        f"{timestamp}"
    )
    return name


# -----------------------------------------------------------
# Save model + test data + predictions + history + config
# -----------------------------------------------------------
def save_model_package(
    model,
    model_name,
    x_test,
    y_test,
    x_train,
    y_train,
    y_pred_train,
    y_pred_test,
    history=None,
    config=None,
    base_dir="models"
):
    """
    Saves all model artifacts into a folder:
        model.h5
        x_test.npy
        y_test.npy
        y_pred.npy
        history.json (optional)
        config.json (optional)
    """

    # Full path for this model
    save_dir = os.path.join(base_dir, model_name)

    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)

    # 1. Save model
    model.save(f"{save_dir}/model.h5")
    print(f"[OK] Saved model: {save_dir}/model.h5")

    # 2. Save arrays
    np.save(f"{save_dir}/x_test.npy", x_test)
    np.save(f"{save_dir}/y_test.npy", y_test)
   # np.save(f"{save_dir}/x_train.npy", x_train)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/y_pred_train.npy", y_pred_train)
    np.save(f"{save_dir}/y_pred_test.npy", y_pred_test)
    print(f"[OK] Saved test arrays: x_test, y_test, x_train, y_train,y_pred_train, y_pred_test")

    print(f"[DONE] All model artifacts saved under folder: {save_dir}")
    

import numpy as np

import numpy as np

def select_representative_samples(
    x,
    y_true,
    y_pred,
    percentile=75,
    k=1,
    return_indices=True
):
    """
    Select representative samples at a given percentile of closeness
    to the class target (0 or 1), WITHOUT enforcing correct prediction.

    For y_true = 1 → target = 1
    For y_true = 0 → target = 0

    Percentile is computed on |y_pred - target| within each true class.

    Parameters
    ----------
    x : np.ndarray
        Input samples (x_test)
    y_true : array-like
        True labels (0/1)
    y_pred : array-like
        Predicted probabilities for class 1
    percentile : int (0–100)
        Percentile of distance-to-target to select
    k : int
        Number of samples per class
    return_indices : bool
        Whether to return indices

    Returns
    -------
    dict
        {
          'y1': {...},
          'y0': {...}
        }
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def _select_for_class(cls, target):
        mask = (y_true == cls)
        idx_all = np.where(mask)[0]

        if len(idx_all) == 0:
            return None

        dist = np.abs(y_pred[idx_all] - target)

        ref_dist = np.percentile(dist, percentile)

        order = np.argsort(np.abs(dist - ref_dist))
        idx_sel = idx_all[order[:k]]

        return {
            "idx": idx_sel if return_indices else None,
            "y_true": y_true[idx_sel],
            "y_pred": y_pred[idx_sel]
        }

    return {
        "y1": _select_for_class(cls=1, target=1.0),
        "y0": _select_for_class(cls=0, target=0.0),
    }


def select_representative_samples_top(
    x,
    y_true,
    y_pred,
    threshold=0.5,
    k=1,
    return_indices=True
):
    """
    Select representative correctly-predicted samples:
      - closest to 1 (y=1)
      - closest to 0 (y=0)
      - closest to 0.5 (max uncertainty)

    Parameters
    ----------
    x : np.ndarray
        Test input data (aligned with y_true, y_pred)
    y_true : array-like
        True labels (0/1)
    y_pred : array-like
        Predicted probabilities for class 1
    threshold : float, default=0.5
        Classification threshold
    k : int, default=1
        Number of samples to return per category
    return_indices : bool, default=True
        Whether to return indices in addition to x

    Returns
    -------
    dict
        {
          'closest_to_1': {'idx', 'x', 'y_true', 'y_pred'},
          'closest_to_0': {...},
          'closest_to_05': {...}
        }
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Correct predictions
    y_hat = (y_pred >= threshold).astype(int)
    correct = (y_hat == y_true)

    def _select(mask, target):
        idx_all = np.where(mask)[0]
        if len(idx_all) == 0:
            return None

        dist = np.abs(y_pred[idx_all] - target)
        topk = idx_all[np.argsort(dist)[:k]]

        return {
            "idx": topk if return_indices else None,
            "x": x[topk],
            "dist": dist,
            "topk": topk,
            "y_true": y_true[topk],
            "y_pred": y_pred[topk],
        }

    return {
        "closest_to_1":  _select(correct & (y_true == 1), 1.0),
        "closest_to_0":  _select(correct & (y_true == 0), 0.0),
        "closest_to_05": _select(correct, 0.5),
    }

