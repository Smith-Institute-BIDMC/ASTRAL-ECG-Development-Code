import os
import json
import datetime
import numpy as np

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
 
    save_dir = os.path.join(base_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    model.save(f"{save_dir}/model.h5")
    print(f"[OK] Saved model: {save_dir}/model.h5")

    np.save(f"{save_dir}/x_test.npy", x_test)
    np.save(f"{save_dir}/y_test.npy", y_test)
    np.save(f"{save_dir}/x_train.npy", x_train)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/y_pred_train.npy", y_pred_train)
    np.save(f"{save_dir}/y_pred_test.npy", y_pred_test)
    print(f"[OK] Saved arrays: x_test, y_test, x_train, y_train,y_pred_train, y_pred_test")

    print(f"[DONE] All saved under folder: {save_dir}")
    
def select_representative_samples(
    x,
    y_true,
    y_pred,
    percentile=75,
    k=1,
    return_indices=True
):
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
