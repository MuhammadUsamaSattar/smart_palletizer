import argparse
from collections import defaultdict
from collections.abc import Callable
import csv
from datetime import datetime
import gc
import glob
import itertools
import math
import os
from tqdm import tqdm
from typing import Tuple, List

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from smart_palletizer_cnn import yolo_model
from smart_palletizer_cnn.box_and_pose_dataset import BoxAndPoseDataset
from smart_palletizer_utils import utils
import torch
from torch.amp import autocast, GradScaler
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.ops import box_iou


def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create batches from samples (X, y).

    Args:
        batch (List[torch.Tensor]): Batches a list containing X and y.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two separate tensors containing batches of X and y.
    """
    X_batch = torch.stack([item[0] for item in batch], dim=0)
    y_batch = [item[1] for item in batch]  # keep labels as list of dicts
    return X_batch, y_batch


def generate_anchors(scales: list, ratios: list) -> torch.Tensor:
    """Generates anchor boxes for given scales and aspect ratios.

    Args:
        scales (list): List of scales.
        ratios (list): List of ratios.

    Returns:
        torch.Tensor: Tensor containing unque combinations of scales and ratios.
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * math.sqrt(ratio)
            height = scale / math.sqrt(ratio)
            anchors.append((width, height))
    return torch.tensor(anchors)


def get_loss_fn(
    lambda_coords: int | float,
    lambda_no_obj: int | float,
    grid_size: int,
    anchors: torch.Tensor,
    num_classes: int,
    device: str,
) -> Callable:
    """Initializes the parameters for loss function and reutrns a callable loss function.

    Args:
        lambda_coords (int | float): Weightage of bounding box dimensions in loss calculation.
        lambda_no_obj (int | float): Weightage of loss penalty for predicting a bounding box where none exists.
        grid_size (int): Size of the final grid at which predictions are done.
        anchors (torch.Tensor): Tensor of anchors scales and sizes.
        num_classes (int): Number of prediction classes.
        device (str): Device.

    Returns:
        function: Function that calculates the loss value.
    """

    def get_loss_val(pred: torch.Tensor, labels: List[dict]) -> torch.Tensor:
        labels = encode_labels_to_YOLO_format(
            labels, grid_size, anchors, num_classes, device
        )
        """Calculates the loss value.

        Returns:
            torch.Tensor: Loss value.
        """

        obj_mask = labels[..., 4] == 1.0
        no_obj_mask = labels[..., 4] == 0.0

        num_objs = obj_mask.sum().clamp(min=1)
        num_no_objs = no_obj_mask.sum().clamp(min=1)

        loss_bbox_coords = (
            lambda_coords
            * torch.sum((torch.sigmoid(pred[obj_mask, :2]) - labels[obj_mask, :2]) ** 2)
            / num_objs
        )
        loss_bbox_scale = (
            lambda_coords
            * torch.sum((pred[obj_mask, 2:4] - labels[obj_mask, 2:4]) ** 2)
            / num_objs
        )

        loss_obj = (
            binary_cross_entropy_with_logits(
                pred[obj_mask, 4], labels[obj_mask, 4], reduction="sum"
            )
            / num_objs
        )
        loss_no_obj = (
            lambda_no_obj
            * binary_cross_entropy_with_logits(
                pred[no_obj_mask, 4], labels[no_obj_mask, 4], reduction="sum"
            )
            / num_no_objs
        )

        loss_classification = (
            cross_entropy(pred[obj_mask, 5:], labels[obj_mask, 5:], reduction="sum")
            / num_objs
        )

        total_loss = (
            loss_bbox_coords
            + loss_bbox_scale
            + loss_obj
            + loss_no_obj
            + loss_classification
        )

        return total_loss

    return get_loss_val


def encode_labels_to_YOLO_format(
    labels: List[dict],
    grid_size: int,
    anchors: torch.Tensor,
    num_classes: int,
    device: str,
) -> torch.Tensor:
    """Ecnodes the labels in data to YOLO format.

    Args:
        labels (List[dict]): List of dictionaries containing the labels.
        grid_size (int): Size of the grid at which to predict.
        anchors (torch.Tensor): Sizes and scales of anchors boxes.
        num_classes (int): Number of classes in predictions.
        device (str): Device.

    Returns:
        torch.Tensor: Labels encoded in YOLO format.
    """
    B = len(labels)
    S = grid_size
    A = anchors.shape[0]

    res = torch.zeros((B, S, S, A, 5 + num_classes), device=device)

    for b, label in enumerate(labels):
        bboxes = label["bboxes"].to(device)  # (N, 4)
        classes = label["categories"].float().to(device)  # (N,)

        bboxes[:, :2] = bboxes[:, :2] + (bboxes[:, 2:] / 2)
        cx, cy, w, h = bboxes.T

        # Grid cell indices
        gx = (cx * S).long()
        gy = (cy * S).long()

        # Offsets within cell
        tx = (cx * S) - gx.float()
        ty = (cy * S) - gy.float()

        # ---- Anchor matching ----
        # Compute IoU between each box and each anchor (N, A)
        box_wh = torch.stack([w, h], dim=1)[:, None, :]  # (N, 1, 2)
        anchor_wh = anchors[None, :, :]  # (1, A, 2)
        inter = torch.min(box_wh, anchor_wh).prod(dim=2)
        union = box_wh.prod(dim=2) + anchor_wh.prod(dim=2) - inter
        iou = inter / union

        best_anchor = iou.argmax(dim=1)  # (N,)

        # ---- Encoding ----
        res[b, gy, gx, best_anchor, 0] = tx
        res[b, gy, gx, best_anchor, 1] = ty
        res[b, gy, gx, best_anchor, 2] = torch.log(w / anchors[best_anchor, 0])
        res[b, gy, gx, best_anchor, 3] = torch.log(h / anchors[best_anchor, 1])
        res[b, gy, gx, best_anchor, 4] = 1.0
        res[b, gy, gx, best_anchor, 5:] = classes

    return res


def visualize_prediction(pred: torch.Tensor, image: torch.Tensor) -> None:
    """Plot the prediction on the image.

    Args:
        pred (torch.Tensor): Predictions.
        image (torch.Tensor): Image to drawn predictions on.
    """
    image = image.cpu().detach()

    labels = {"bboxes": pred.cpu().detach()}
    utils.visualize_box_and_pose_data(image, labels, options={"color", "bboxes"})
    del labels


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    grid_size: torch.Tensor,
    iou_threshold_NMS: float,
    iou_threshold_mAP: float | List[float] | Tuple[float],
    desc: str = "Validation",
) -> float:
    """Evaluate the models.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Dataloader to evaluate the model on.
        grid_size (torch.Tensor): Size of the grid at which predictions are made.
        iou_threshold_NMS (float): Threshold of non maximum suppresion.
        iou_threshold_mAP (float | List[float] | Tuple[float]): Threshold(s) for calculating mAP value.
        desc (str, optional): Indicates the type of dataloader to print with tqdm. Defaults to "Validation".

    Raises:
        ValueError: Raised when iou_threshold_mAP is not in the correct format.

    Returns:
        float: mAP value.
    """
    model.eval()
    if isinstance(iou_threshold_mAP, float):
        iou_thresholds = [iou_threshold_mAP]

    elif isinstance(iou_threshold_mAP, (list, tuple)):
        start, end, step = iou_threshold_mAP
        iou_thresholds = torch.arange(start, end + 1e-6, step).tolist()

    else:
        raise ValueError("iou_threshold_mAP must be float or [start, end, step]")

    with torch.inference_mode():
        decoded_preds = []
        labels = []
        for X, y in tqdm(
            data_loader, desc=desc + " Batches", unit="batch", leave=False
        ):
            X = X.to(device)
            pred = model(X)
            decoded_preds.extend(
                yolo_model.YOLOModel.decode_YOLO_encodings(
                    pred, grid_size, anchors, 0.0, iou_threshold_NMS
                )
            )
            labels.extend(y)
        mAP = compute_mAP(decoded_preds, labels, iou_thresholds, device)

        return mAP


def compute_mAP(
    preds: torch.Tensor, labels: List[dict], iou_thresholds: float, device: str
) -> float:
    """Compute the mAP value.

    Args:
        preds (torch.Tensor): Predictions.
        labels (List[dict]): True labels.
        iou_thresholds (float): IOU threshold for mAP calculation.
        device (str): Device.

    Returns:
        float: mAP value.
    """
    if isinstance(iou_thresholds, (float, int)):
        iou_thresholds = torch.tensor([iou_thresholds], device=device)
        return_scalar = True
    else:
        iou_thresholds = torch.tensor(iou_thresholds, device=device)
        return_scalar = False

    num_classes = preds[0].shape[1] - 5
    num_imgs = len(preds)
    aps = torch.zeros(len(iou_thresholds), device=device)

    # Precompute GT counts per image per class
    gt_counts = [lbl["categories"].sum(dim=0).to(device) for lbl in labels]

    for cls in range(num_classes):
        total_gt = sum(g[cls].item() for g in gt_counts)
        if total_gt == 0:
            continue

        confs, ious_all, gt_ids, img_ids = [], [], [], []

        max_gt = max(g[cls].item() for g in gt_counts)

        for img_id, (p, lbl) in enumerate(zip(preds, labels)):
            scores = p[:, 4] * p[:, 5 + cls]
            keep = p[:, 5 + cls] > 0.5

            p = p[keep]
            scores = scores[keep]

            if len(p) == 0:
                continue

            gt = lbl["bboxes"][lbl["categories"][:, cls] == 1].to(device)

            confs.append(scores)
            img_ids.append(torch.full((len(p),), img_id, device=device))

            if len(gt) == 0:
                ious_all.append(torch.zeros(len(p), device=device))
                gt_ids.append(torch.full((len(p),), -1, device=device))
            else:
                ious = box_iou(p[:, :4], gt, fmt="xywh")
                mi, gi = ious.max(dim=1)
                ious_all.append(mi)
                gt_ids.append(gi)

        confs = torch.cat(confs)
        ious_all = torch.cat(ious_all)
        gt_ids = torch.cat(gt_ids)
        img_ids = torch.cat(img_ids)

        order = torch.argsort(confs, descending=True)
        ious_all = ious_all[order]
        gt_ids = gt_ids[order]
        img_ids = img_ids[order]

        flat_gt = img_ids * max_gt + gt_ids
        flat_gt[gt_ids < 0] = -1

        for t_i, t in enumerate(iou_thresholds):
            valid = ious_all >= t
            seen = torch.zeros(num_imgs * max_gt, device=device, dtype=torch.bool)
            tp = torch.zeros(len(valid), device=device)

            for i in torch.nonzero(valid).squeeze(1):
                fg = flat_gt[i]
                if fg >= 0 and not seen[fg]:
                    tp[i] = 1
                    seen[fg] = True

            fp = 1 - tp
            tp_cum = torch.cumsum(tp, 0)
            fp_cum = torch.cumsum(fp, 0)

            recalls = tp_cum / total_gt
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

            precisions = torch.flip(
                torch.cummax(torch.flip(precisions, [0]), dim=0)[0], [0]
            )

            aps[t_i] += torch.trapz(precisions, recalls)

    aps /= num_classes
    return aps.item() if return_scalar else aps.mean().item()


def train_with_hyperparameter_grid_search(
    param_sets: List[List],
    training_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    testing_dataloader: torch.utils.data.DataLoader,
    backbone_in_channels: int,
    prediction_head_num_classes: int,
    image_size: int,
    anchors: torch.Tensor,
    num_epochs: int,
    device: str,
    accumulate_steps: int,
    use_amp: bool,
    print_info_after_batches: bool = False,
    print_info_after_epoch: bool = False,
    visualize_after_batches: bool = False,
    visualize_after_epoch: bool = False,
    visualize_after_training: bool = False,
    iou_threshold_NMS: float = 0.30,
    iou_threshold_mAP: float = 0.50,
    drop_models_after_epochs: int = 5,
    models_to_drop: float = 2 / 3,
    n_remaining_models: int = 5,
    max_patience: int = 5,
    min_delta: float = 0.001,
    verbose: bool = False,
    **kwargs,
) -> None:
    """Train models with various hyperparameter sets.

    Args:
        param_sets (List[List]): Lists of parameters.
        training_dataloader (torch.utils.data.DataLoader): Training dataloader.
        validation_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        testing_dataloader (torch.utils.data.DataLoader): Testing dataloader.
        backbone_in_channels (int): Number of channels in the input data.
        prediction_head_num_classes (int): Number of classes in the prediciton.
        image_size (int): Size of the image.
        anchors (torch.Tensor): Anchors of various scales and sizes.
        num_epochs (int): Number of epochs to train for.
        device (str): Device.
        accumulate_steps (int): Number of training steps to accumulate the gradients for. Useful if desired size of batch does not fit the device.
        use_amp (bool): Whether to use automatic mixed preicision.
        print_info_after_batches (bool, optional): Print training metrics after each batch. Defaults to False.
        print_info_after_epoch (bool, optional): Print training metrics after each epoch. Defaults to False.
        visualize_after_batches (bool, optional): Visualize prediction after each batch. Defaults to False.
        visualize_after_epoch (bool, optional): Visualize prediction after each epoch. Defaults to False.
        visualize_after_training (bool, optional): Visualize prediction after training. Defaults to False.
        iou_threshold_NMS (float, optional): IOU threshold for non maximum suppresion. Defaults to 0.30.
        iou_threshold_mAP (float, optional): IOU threshold(s) for mAP calculation. Defaults to 0.50.
        drop_models_after_epochs (int, optional): Epochs after which models are dropped. Defaults to 5.
        models_to_drop (float, optional): Fraction of models to drop. Defaults to 2/3.
        n_remaining_models (int, optional): Minimum number of models to keep alive. Defaults to 5.
        max_patience (int, optional): Patience for early stopping. Defaults to 5.
        min_delta (float, optional): Delta to measure whether model improved. Defaults to 0.001.
        verbose (bool, optional): Whehter to print extra diagnostic information. Defaults to False.
    """

    models_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "models",
        f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}",
    )
    os.makedirs(
        models_dir,
        exist_ok=True,
    )
    runs = create_models(
        backbone_in_channels,
        prediction_head_num_classes,
        image_size,
        anchors,
        device,
        verbose,
        param_sets,
        models_dir,
    )

    csv_path = os.path.join(models_dir, "data.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([run["run_id"] for run in runs])
        csv_writer.writerow([run["model_name"] for run in runs])
        csv_file.flush()

        n_active_models = len(runs)
        for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
            tqdm.write("################################")
            tqdm.write(
                f"Epochs {epoch+1}/{num_epochs} - Active Models {n_active_models}/{len(runs)}"
            )
            tqdm.write("--------------------------------")
            for run in tqdm(
                runs[:n_active_models], desc="Active Models", unit="model", leave=False
            ):
                if not run["active"]:
                    continue

                X, pred = train_checkpoint_for_one_epoch(
                    training_dataloader,
                    validation_dataloader,
                    backbone_in_channels,
                    prediction_head_num_classes,
                    anchors,
                    num_epochs,
                    device,
                    print_info_after_batches,
                    print_info_after_epoch,
                    visualize_after_batches,
                    visualize_after_epoch,
                    iou_threshold_NMS,
                    iou_threshold_mAP,
                    max_patience,
                    min_delta,
                    verbose,
                    epoch,
                    run,
                    models_dir,
                    accumulate_steps,
                    use_amp,
                )

            runs[:n_active_models] = sorted(
                runs[:n_active_models], key=lambda run: run["mAP"], reverse=True
            )

            if verbose:
                tqdm.write(
                    "lr=learning rate",
                    "lmbd-co=lambda_coords",
                    "lmbd-no=lambda_no_obj",
                    "bb-ndscb=backbone_num_downsampling_conv_blocks",
                    "bb-nndscb=backbone_num_nondownsampling_conv_blocks",
                    "bb-flc=backbone_first_layer_out_channels",
                    "bb-ks=backbone_kernel_size",
                    sep="\n",
                    end="\n\n",
                )

            for run in runs[:n_active_models]:
                if run["active"]:
                    tqdm.write(
                        f"run_id={run["run_id"]}, model_name={run["model_name"]}, mAP@{iou_threshold_mAP}={run["mAP"]:.5f}, Active"
                    )
                else:
                    tqdm.write(
                        f"run_id={run["run_id"]}, model_name={run["model_name"]}, mAP@{iou_threshold_mAP}={run["mAP"]:.5f}, Early Stopped"
                    )

            csv_writer.writerow(
                [
                    run["mAP"] if run["active"] else None
                    for run in sorted(runs, key=lambda run: run["run_id"])
                ]
            )
            csv_file.flush()

            if (
                n_active_models > n_remaining_models
                and (epoch + 1) % drop_models_after_epochs == 0
            ):
                tqdm.write(f"Dropping {models_to_drop:.0%} of worst active models!")
                n_active_models = drop_runs(
                    models_to_drop,
                    n_remaining_models,
                    runs,
                    n_active_models,
                    models_dir,
                )

    for filename in glob.glob(models_dir + "/curr_*"):
        os.remove(filename)

    print("################################")
    for i in range(n_active_models):
        model = yolo_model.YOLOModel(
            backbone_num_downsampling_conv_blocks=runs[i][
                "backbone_num_downsampling_conv_blocks"
            ],
            backbone_num_nondownsampling_conv_blocks=runs[i][
                "backbone_num_nondownsampling_conv_blocks"
            ],
            backbone_in_channels=backbone_in_channels,
            backbone_first_layer_out_channels=runs[i][
                "backbone_first_layer_out_channels"
            ],
            backbone_kernel_size=runs[i]["backbone_kernel_size"],
            prediction_head_num_anchors=anchors.shape[0],
            prediction_head_num_classes=prediction_head_num_classes,
        )
        model, _, _, _, _ = load_model(
            model,
            models_dir,
            runs[i]["model_name"],
            None,
            best=True,
            device=device,
            verbose=verbose,
        )
        print(
            f"run_id={runs[i]['run_id']}, "
            f"model_name={runs[i]['model_name']}, "
            f"Test mAP@{iou_threshold_mAP}={evaluate_model(
                model,
                testing_dataloader,
                runs[i]['grid_size'],
                iou_threshold_NMS,
                iou_threshold_mAP,
                desc='Testing'
            ):.5f}"
        )
    print("################################")

    if visualize_after_training:
        visualize_prediction(pred, X)

    plot_hyperparam_search_summary(csv_path)


def drop_runs(
    models_to_drop: float,
    n_remaining_models: int,
    runs: List[dict],
    n_active_models: int,
    models_dir: str,
) -> int:
    """Drops some active models.

    Args:
        models_to_drop (float): Fraction of models to drop.
        n_remaining_models (int): Minimum number of active models.
        runs (List[dict]): List of runs.
        n_active_models (int): Number of current active models.
        models_dir (str): Directory where models are stored.

    Returns:
        int: Updated number of active models.
    """
    prev_n_active_models = n_active_models
    n_active_models = max(
        math.floor(n_active_models * (1 - models_to_drop)), n_remaining_models
    )
    for run in runs[n_active_models:prev_n_active_models]:
        run["active"] = False
        if run["patience"] > 0:
            os.remove(os.path.join(models_dir, "curr_" + run["model_name"]))

    return n_active_models


def train_checkpoint_for_one_epoch(
    training_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    backbone_in_channels: int,
    prediction_head_num_classes: int,
    anchors: torch.Tensor,
    num_epochs: int,
    device: str,
    print_info_after_batches: bool,
    print_info_after_epoch: bool,
    visualize_after_batches: bool,
    visualize_after_epoch: bool,
    iou_threshold_NMS: float,
    iou_threshold_mAP: List[float],
    max_patience: int,
    min_delta: float,
    verbose: bool,
    epoch: int,
    run: dict,
    models_dir: str,
    accumulate_steps: int,
    use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train a model for one epoch.

    Args:
        training_dataloader (torch.utils.data.DataLoader): Training dataloader.
        validation_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        backbone_in_channels (int): Number of channels in the input.
        prediction_head_num_classes (int): Number of classes in prediction.
        anchors (torch.Tensor): Tensor of anchors of various scales and ratios.
        num_epochs (int): Number of epochs.
        device (str): Device.
        print_info_after_batches (bool): Print training information after each batch.
        print_info_after_epoch (bool): Print training information after each epoch.
        visualize_after_batches (bool): Visualize prediction after each batch.
        visualize_after_epoch (bool): Visualize prediction after each epoch.
        iou_threshold_NMS (float): IOU threshold for non maximum suppression.
        iou_threshold_mAP (List[float]): IOU thershold(s) for mAP calculation.
        max_patience (int): Maximum patience for early stopping.
        min_delta (float): Minimum delta before which model is considered to have improved.
        verbose (bool): Verbosity for printing extra diagnostic information.
        epoch (int): Current epoch.
        run (dict): List of runs.
        models_dir (str): Directory containing saved models.
        accumulate_steps (int): Number of steps for which to accumulate gradients. Useful when desired batch size does not fit on the device.
        use_amp (bool): Whether to use Automatic Mixed Precision.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of one image and its corresponding prediction.
    """
    if verbose:
        tqdm.write(f"Processing run_id={run["run_id"]}")

    model = yolo_model.YOLOModel(
        backbone_num_downsampling_conv_blocks=run[
            "backbone_num_downsampling_conv_blocks"
        ],
        backbone_num_nondownsampling_conv_blocks=run[
            "backbone_num_nondownsampling_conv_blocks"
        ],
        backbone_in_channels=backbone_in_channels,
        backbone_first_layer_out_channels=run["backbone_first_layer_out_channels"],
        backbone_kernel_size=run["backbone_kernel_size"],
        prediction_head_num_anchors=anchors.shape[0],
        prediction_head_num_classes=prediction_head_num_classes,
    )
    optimizer = Adam(model.parameters(), lr=run["lr"])
    model, optimizer, _, _, _ = load_model(
        model,
        models_dir,
        run["model_name"],
        optimizer,
        best=run["patience"] == 0,
        device=device,
        verbose=verbose,
    )

    loss_fn = get_loss_fn(
        lambda_coords=run["lambda_coords"],
        lambda_no_obj=run["lambda_no_obj"],
        grid_size=run["grid_size"],
        anchors=anchors,
        num_classes=prediction_head_num_classes,
        device=device,
    )

    mAP, loss, X, pred = train_one_epoch(
        training_dataloader,
        validation_dataloader,
        num_epochs,
        device,
        print_info_after_batches,
        print_info_after_epoch,
        visualize_after_batches,
        visualize_after_epoch,
        iou_threshold_NMS,
        iou_threshold_mAP,
        run["grid_size"],
        model,
        optimizer,
        loss_fn,
        epoch,
        accumulate_steps=accumulate_steps,
        use_amp=use_amp,
    )

    run["mAP"] = mAP
    if run["mAP"] >= run["best_mAP"] + min_delta:
        run["best_mAP"] = run["mAP"]

        if run["patience"] > 0:
            os.remove(os.path.join(models_dir, "curr_" + run["model_name"]))
        run["patience"] = 0
    else:
        run["patience"] += 1
        if run["patience"] >= max_patience:
            run["active"] = False
            os.remove(os.path.join(models_dir, "curr_" + run["model_name"]))

    save_model(
        model=model,
        models_dir=models_dir,
        model_name=run["model_name"],
        optimizer=optimizer,
        mAP=mAP,
        loss=loss,
        epoch=epoch,
        best=run["patience"] == 0,
        verbose=verbose,
    )
    free_memory(model, optimizer)

    return X[0], pred[0]


def create_models(
    backbone_in_channels: int,
    prediction_head_num_classes: int,
    image_size: int,
    anchors: torch.Tensor,
    device: str,
    verbose: bool,
    param_sets: List[List],
    models_dir: str,
) -> List[dict]:
    """Creates the models on the disk.

    Args:
        backbone_in_channels (int): Number of channels in the input.
        prediction_head_num_classes (int): Number of classes in the prediction.
        image_size (int): Size of the image.
        anchors (torch.Tensor): Tensor of anchors of various sizes and ratios.
        device (str): Device.
        verbose (bool): Verbosity for printing extra diagnostic information.
        param_sets (List[List]): Sets of parameters over which the hyperparameter search is being done.
        models_dir (str): Directory where the models should be saved.

    Returns:
        List[dict]: List of runs.
    """
    runs = []
    for i, params in enumerate(
        tqdm(list(param_sets), desc="Creating Models", unit="model", leave=False)
    ):
        if verbose:
            tqdm.write(f"Creating run_id={i}")

        runs.append({})
        runs[-1]["run_id"] = i
        runs[-1]["backbone_num_downsampling_conv_blocks"] = params[0]
        runs[-1]["backbone_num_nondownsampling_conv_blocks"] = params[1]
        runs[-1]["backbone_first_layer_out_channels"] = params[2]
        runs[-1]["backbone_kernel_size"] = params[3]
        runs[-1]["lambda_coords"] = params[4]
        runs[-1]["lambda_no_obj"] = params[5]
        runs[-1]["lr"] = params[6]
        runs[-1]["grid_size"] = image_size // (2 ** params[0])
        runs[-1]["mAP"] = -1.0
        runs[-1]["best_mAP"] = -1.0
        runs[-1]["patience"] = 0
        runs[-1]["active"] = True

        model = yolo_model.YOLOModel(
            backbone_num_downsampling_conv_blocks=runs[-1][
                "backbone_num_downsampling_conv_blocks"
            ],
            backbone_num_nondownsampling_conv_blocks=runs[-1][
                "backbone_num_nondownsampling_conv_blocks"
            ],
            backbone_in_channels=backbone_in_channels,
            backbone_first_layer_out_channels=runs[-1][
                "backbone_first_layer_out_channels"
            ],
            backbone_kernel_size=runs[-1]["backbone_kernel_size"],
            prediction_head_num_anchors=anchors.shape[0],
            prediction_head_num_classes=prediction_head_num_classes,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=runs[-1]["lr"])

        runs[-1]["model_name"] = "_".join(
            [
                model._get_name(),
                f"lr-{runs[-1]["lr"]}",
                f"lmbd-co-{runs[-1]["lambda_coords"]}",
                f"lmbd-no-{runs[-1]["lambda_no_obj"]}",
                f"bb-ndscb-{runs[-1]["backbone_num_downsampling_conv_blocks"]}",
                f"bb-nndscb-{runs[-1]["backbone_num_nondownsampling_conv_blocks"]}",
                f"bb-flc-{runs[-1]["backbone_first_layer_out_channels"]}",
                f"bb-ks-{runs[-1]["backbone_kernel_size"]}",
            ]
        )

        save_model(
            model=model,
            models_dir=models_dir,
            model_name=runs[-1]["model_name"],
            optimizer=optimizer,
            mAP=-1,
            loss=-1,
            epoch=0,
            verbose=verbose,
        )

        free_memory(model, optimizer)

    return runs


def free_memory(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Frees the memory from the model and optimizer.

    Args:
        model (torch.nn.Module): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
    """
    if model is not None:
        model = model.to("cpu")  # move weights off GPU
    if optimizer is not None:
        optimizer.state.clear()  # drop optimizer tensors

    del optimizer
    del model

    gc.collect()
    torch.cuda.empty_cache()


def save_model(
    model: torch.nn.Module,
    models_dir: str,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    mAP: float,
    loss: float,
    epoch: int,
    best: bool = True,
    verbose: bool = False,
) -> None:
    """Save the model on the disk.

    Args:
        model (torch.nn.Module): Model.
        models_dir (str): Direcotry where to save the model.
        model_name (str): Name of the model.
        optimizer (torch.optim.Optimizer): Optimizer.
        mAP (float): mAP value.
        loss (float): Loss value.
        epoch (int): Number of current epoch.
        best (bool, optional): Indicates if this is the best version of this model. If not, then the model is saved with a different name. Defaults to True.
        verbose (bool, optional): Verbosity for printing extra diagnostic information. Defaults to False.
    """
    if not best:
        model_name = "curr_" + model_name
    os.makedirs(
        models_dir,
        exist_ok=True,
    )
    model_path = os.path.join(
        models_dir,
        model_name,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mAP": mAP,
            "epoch": epoch,
            "loss": loss,
        },
        model_path,
    )

    if verbose:
        tqdm.write(f"Model saved to {model_path}")


def load_model(
    model: torch.nn.Module,
    models_dir: str,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    device: str,
    best: bool = True,
    verbose: bool = False,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, float, float, int]:
    """Load the model from disk.

    Args:
        model (torch.nn.Module): Model.
        models_dir (str): Direcotry where to save the model.
        model_name (str): Name of the model.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device.
        best (bool, optional): Indicates if this is the best version of this model. If not, then the model is saved with a different name. Defaults to True.
        verbose (bool, optional): Verbosity for printing extra diagnostic information. Defaults to False.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, float, float, int]: Tuple containing model, optimizer, mAP, loss and epoch.
    """
    if not best:
        model_name = "curr_" + model_name

    model_path = os.path.join(
        models_dir,
        model_name,
    )
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    mAP = checkpoint["mAP"]
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"]

    del checkpoint
    if verbose:
        tqdm.write(f"Model loaded from {model_path}")

    return model, optimizer, mAP, loss, epoch


def train_one_epoch(
    training_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: str,
    print_info_after_batches: bool,
    print_info_after_epoch: bool,
    visualize_after_batches: bool,
    visualize_after_epoch: bool,
    iou_threshold_NMS: float,
    iou_threshold_mAP: List[float],
    grid_size: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epoch: int,
    accumulate_steps: int,
    use_amp: bool,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    """Train the model for one epoch.

    Args:
        training_dataloader (torch.utils.data.DataLoader): Training dataloader.
        validation_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        num_epochs (int): Number of epochs.
        device (str): Device.
        print_info_after_batches (bool): Print training information after each batch.
        print_info_after_epoch (bool): Print training information after each epoch.
        visualize_after_batches (bool): Visualize prediction after each batch.
        visualize_after_epoch (bool): Visualize prediction after each epoch.
        iou_threshold_NMS (float): IOU threshold for non maximum suppression.
        iou_threshold_mAP (List[float]): IOU thershold(s) for mAP calculation.
        grid_size (int): Size of the grid on which prediction is made.
        model (torch.nn.Module): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (function): Loss function.
        epoch (int): Current epoch.
        accumulate_steps (int): Number of steps for which to accumulate gradients. Useful when desired batch size does not fit on the device.
        use_amp (bool): Whether to use Automatic Mixed Precision.

    Returns:
        Tuple[float, float, torch.Tensor, torch.Tensor]: Tuple containing mAP, average epoch loss and, an image from the training dataset and its corresponding prediction.
    """
    model.train()
    epoch_loss = 0

    scaler = GradScaler(device=device, enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for i, (X, y) in enumerate(
        tqdm(training_dataloader, desc="Training Batches", unit="batch", leave=False)
    ):
        X, y = X.to(device), y

        with autocast(device_type=device, enabled=use_amp):
            pred = model(X)

        with autocast(device_type=device, enabled=False):
            raw_loss = loss_fn(pred=pred, labels=y)
            loss = raw_loss / accumulate_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulate_steps == 0 or (i + 1) == len(training_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += raw_loss.item()

        if i % max(len(training_dataloader) // 10, 1) == 0:
            if print_info_after_batches:
                tqdm.write(f"Batch {i+1}/{len(training_dataloader)}, {loss.item()=}")
            if visualize_after_batches:
                visualize_prediction(pred[0], X[0])

    mAP = evaluate_model(
        model,
        validation_dataloader,
        grid_size,
        iou_threshold_NMS,
        iou_threshold_mAP,
        "Validation",
    )

    if print_info_after_epoch:
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs}, Average epoch loss = {epoch_loss/len(training_dataloader)}, mAP@{iou_threshold_mAP}={mAP}"
        )

    if visualize_after_epoch:
        visualize_prediction(pred[0], X[0])
    return mAP, epoch_loss / len(training_dataloader), X[0], pred[0]


def plot_hyperparam_search_summary(
    csv_path: str, top_k: int = 20, interaction_pairs: list | None = None
) -> None:
    """Plot the summary of the hyperparameter search.

    Args:
        csv_path (str): Path of the csv file containing mAP values for each model over epochs.
        top_k (int, optional): Number of models to show in the top performance charts. Defaults to 20.
        interaction_pairs (list | None, optional): List of parameters whose interactions to show. Defaults to None. None shows all interactions.
    """
    # ==========================================================
    # 1. LOAD + PREPROCESS
    # ==========================================================
    df = pd.read_csv(csv_path, header=None)
    model_names = df.iloc[1].values
    values = df.iloc[2:].apply(pd.to_numeric, errors="coerce")

    n_models = len(model_names)
    run_ids = np.arange(n_models)

    max_mAP = values.max(axis=0)
    mean_mAP = values.mean(axis=0)
    survival_len = values.notna().sum(axis=0)

    model_with_mAP = list(zip(model_names, max_mAP.values))
    model_with_mAP.sort(key=lambda name_val_pair: name_val_pair[1], reverse=True)

    for model_name, mAP in model_with_mAP:
        print(f"{model_name:105} mAP={float(mAP):1.5}")

    # ----------------------------------------------------------
    # Parse hyperparameters from model names
    # ----------------------------------------------------------
    parsed_params = []
    hyperparams = defaultdict(list)

    for name in model_names:
        params = {}
        for part in name.split("_"):
            if "-" in part:
                *k, v = part.split("-", -1)
                key = "-".join(k)
                try:
                    params[key] = float(v)
                except ValueError:
                    params[key] = v
        parsed_params.append(params)
        for k, v in params.items():
            hyperparams[k].append(v)

    hp_df = pd.DataFrame(parsed_params)
    hp_df["max_mAP"] = max_mAP.values
    hp_df["epochs"] = survival_len.values

    # ==========================================================
    # 2. FIGURE 1: SUMMARY + HYPERPARAM EFFECTS
    # ==========================================================
    fig1 = plt.figure(figsize=(16, 12))
    gs1 = GridSpec(3, 4, figure=fig1)
    plot_idx = 0

    def next_ax():
        nonlocal plot_idx
        row, col = divmod(plot_idx, 4)
        ax = fig1.add_subplot(gs1[row, col])
        plot_idx += 1
        return ax

    # --- Top-K leaderboard
    ax = next_ax()
    top_idx = max_mAP.sort_values(ascending=False).head(top_k).index
    ax.barh(range(len(top_idx)), max_mAP[top_idx][::-1])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([f"run_{i}" for i in top_idx][::-1])
    ax.set_title(f"Top {top_k} Models (Max mAP)")
    ax.set_xlabel("mAP")

    # --- Distribution
    ax = next_ax()
    ax.hist(max_mAP.dropna(), bins=30)
    ax.set_title("Distribution of Max mAP")

    # --- Survival
    ax = next_ax()
    ax.scatter(run_ids, survival_len, s=10)
    ax.set_title("Epochs Survived")
    ax.set_xlabel("Run ID")

    # --- Best mAP vs run_id
    ax = next_ax()
    ax.scatter(run_ids, max_mAP, s=10)
    ax.set_title("Best mAP vs Run ID")
    ax.set_xlabel("Run ID")

    # --- mAP vs epochs for top_k
    ax = next_ax()
    for i in top_idx:
        y = values.iloc[:, i].dropna().values
        y = np.insert(y, 0, 0.0)
        ax.plot(np.arange(len(y)), y, alpha=0.4)
    ax.set_title(f"Top-{top_k} mAP vs Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")

    # --- Hyperparameter plots
    for hp in hp_df.columns:
        if hp in ["max_mAP", "epochs"]:
            continue
        if plot_idx >= 12:
            break

        ax = next_ax()
        data = hp_df[[hp, "max_mAP"]].dropna()

        # ===============================
        # LEARNING RATE SPECIAL HANDLING
        # ===============================
        if hp == "lr":
            ax.scatter(data[hp], data["max_mAP"], alpha=0.4, s=10)
            ax.set_xscale("log")
            ax.set_xlabel("lr (log scale)")
            ax.set_ylabel("Max mAP")
            ax.set_title("lr vs Max mAP")
            continue

        # ===============================
        # DISCRETE PARAMS → BOXPLOT
        # ===============================
        grouped = data.groupby(hp)["max_mAP"].apply(list)
        if len(grouped) < 2:
            ax.axis("off")
            continue

        ax.boxplot(
            grouped.values, tick_labels=grouped.index.astype(str), showfliers=False
        )
        ax.set_title(hp, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    # Turn off unused axes
    for i in range(plot_idx, 12):
        row, col = divmod(i, 4)
        fig1.add_subplot(gs1[row, col]).axis("off")

    plt.tight_layout()

    # ==========================================================
    # 3. FIGURE 2: INTERACTION HEATMAPS (LR-AWARE)
    # ==========================================================
    if interaction_pairs is None:
        keys = [k for k in hp_df.columns if k not in ["max_mAP", "epochs"]]
        interaction_pairs = list(itertools.combinations(keys, 2))

    fig2 = plt.figure(figsize=(21, 12))
    gs2 = GridSpec(3, 7, figure=fig2)
    plot_idx = 0

    def next_ax2():
        nonlocal plot_idx
        row, col = divmod(plot_idx, 7)
        ax = fig2.add_subplot(gs2[row, col])
        plot_idx += 1
        return ax

    for p1, p2 in interaction_pairs:
        if plot_idx >= 21:
            break

        subset = hp_df[[p1, p2, "max_mAP"]].dropna()
        if subset.empty:
            continue

        # Bin learning rate if involved
        for p in (p1, p2):
            if p == "lr":
                pmin = subset[p].min()
                pmax = subset[p].max()
                # If all LR values are (nearly) identical, pd.cut will fail because
                # bin edges are not unique. Convert the values to a string label
                # so pivoting still works. Otherwise use log-spaced bins and
                # drop duplicate edges.
                if np.isclose(pmin, pmax):
                    subset[p] = subset[p].map(lambda v: f"{v:.1e}")
                else:
                    bins = np.logspace(np.log10(pmin), np.log10(pmax), 8)
                    subset[p] = pd.cut(subset[p], bins, duplicates="drop")

        pivot = subset.pivot_table(
            index=p2,
            columns=p1,
            values="max_mAP",
            aggfunc="mean",
            observed=False,
        )

        ax = next_ax2()
        im = ax.imshow(pivot.values, origin="lower", aspect="auto")
        # --- X axis ticks (handle binned lr correctly)
        if isinstance(pivot.columns[0], pd.Interval):
            x_edges = [iv.left for iv in pivot.columns] + [pivot.columns[-1].right]
            ax.set_xticks(np.arange(len(x_edges)))
            ax.set_xticklabels([f"{v:.1e}" for v in x_edges], rotation=30)
            ax.set_xlim(0, len(x_edges) - 1)
        else:
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns.astype(str), rotation=30)

        # --- Y axis ticks
        if isinstance(pivot.index[0], pd.Interval):
            y_edges = [iv.left for iv in pivot.index] + [pivot.index[-1].right]
            ax.set_yticks(np.arange(len(y_edges)))
            ax.set_yticklabels([f"{v:.1e}" for v in y_edges])
            ax.set_ylim(0, len(y_edges) - 1)
        else:
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index.astype(str))

        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f"{p1} x {p2}")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused axes
    for i in range(plot_idx, 21):
        row, col = divmod(i, 7)
        fig2.add_subplot(gs2[row, col]).axis("off")

    plt.tight_layout()
    plt.show()


def test_model(
    model_path: str,
    data_loader: torch.utils.data.DataLoader,
    image_size: int,
    backbone_in_channels: int,
    anchors: torch.Tensor,
    eval_iou_threshold_mAP: float | List[float] | Tuple[float],
    eval_iou_threshold_NMS: float,
    vis_iou_threshold_NMS: float,
    vis_conf_threshold: float,
    prediction_head_num_classes: int,
    device: str,
    **kwargs,
) -> None:
    """Test the model and visualize preditions.

    Args:
        model_path (str): Path of the model.
        data_loader (torch.utils.data.DataLoader): Dataloader on which to test.
        image_size (int): Size of the images.
        backbone_in_channels (int): Number of channels in the input.
        anchors (torch.Tensor): Tensor of anchors of various shapes and ratios.
        eval_iou_threshold_mAP (float | List[float] | Tuple[float]): IOU threshold for mAP claculation for evaluation.
        eval_iou_threshold_NMS (float): IOU threshold for non maximum suppresion for evaluation.
        vis_iou_threshold_NMS (float): IOU threshold for non maximum suppression for visualizing predictions.
        vis_conf_threshold (float): Threshold for confidence to use for visualizing predictions.
        prediction_head_num_classes (int): Number of classes in the prediction.
        device (str): Device.
    """
    model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        model_path,
    )
    model_dir, model_name = os.path.split(model_path)

    parts = model_name.split("_")[1:]
    params = {}
    for part in parts:
        *k, v = part.split("-", -1)
        k = "-".join(k)
        params[k] = v

    model = yolo_model.YOLOModel(
        backbone_num_downsampling_conv_blocks=int(params["bb-ndscb"]),
        backbone_num_nondownsampling_conv_blocks=int(params["bb-nndscb"]),
        backbone_in_channels=backbone_in_channels,
        backbone_first_layer_out_channels=int(params["bb-flc"]),
        backbone_kernel_size=int(params["bb-ks"]),
        prediction_head_num_anchors=anchors.shape[0],
        prediction_head_num_classes=prediction_head_num_classes,
    )
    model, *_ = load_model(model, model_dir, model_name, None, device)
    model.eval()
    with torch.inference_mode():
        tqdm.write(
            f"mAP@{eval_iou_threshold_mAP}={evaluate_model(model, data_loader, image_size // (2 ** int(params["bb-ndscb"])), eval_iou_threshold_NMS, eval_iou_threshold_mAP, desc="Test")}"
        )
        for X, y in data_loader:
            X = X.to(device)
            preds = model(X)
            preds = yolo_model.YOLOModel.decode_YOLO_encodings(
                preds,
                image_size // (2 ** int(params["bb-ndscb"])),
                anchors,
                vis_conf_threshold,
                vis_iou_threshold_NMS,
            )
            for i in range(len(preds)):
                visualize_prediction(preds[i], X[i])

    return


if __name__ == "__main__":
    params = {
        "backbone_in_channels": 2,
        "prediction_head_num_classes": 2,
        "image_size": 512,
        "num_epochs": 100,
        "batch_size": 32,
        "effective_batch_size": 32,
        "print_info_after_batches": False,
        "print_info_after_epoch": False,
        "visualize_after_batches": False,
        "visualize_after_epoch": False,
        "visualize_after_training": False,
        "iou_threshold_NMS": 0.30,
        "drop_models_after_epochs": 5,
        "models_to_drop": 1 / 6,
        "n_remaining_models": 3,
        "use_amp": True,
    }

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    anchors = generate_anchors([0.10, 0.15, 0.20, 0.25], [0.75, 1.00, 1.25, 1.50]).to(
        device
    )

    data = BoxAndPoseDataset(
        "data/synthetic_data",
        "coco_annotations.json",
        "hdf5",
        transform_feature=True,
        transform_label=True,
        image_size=params["image_size"],
        dtype=torch.float32,
    )

    training_data, validation_data, testing_data = random_split(
        data,
        [0.70, 0.15, 0.15],
    )

    training_dataloader = DataLoader(
        training_data,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    testing_dataloader = DataLoader(
        testing_data,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Parameters in the following order:
    # backbone_num_downsampling_conv_blocks
    # backbone_num_nondownsampling_conv_blocks
    # backbone_first_layer_out_channels
    # backbone_kernel_size
    # lambda_coords
    # lambda_no_obj
    # lr
    param_sets = [
        [3, 10, 32, 5, 12, 0.375, 0.00048],
        [3, 10, 32, 5, 10, 0.375, 0.00048],
        [3, 10, 32, 5, 10, 0.375, 0.00016],
        [3, 10, 32, 5, 10, 0.250, 0.00048],
        [3, 7, 16, 3, 8, 0.375, 0.00048],
        [3, 7, 16, 3, 10, 0.375, 0.00080],
    ]

    params.update(
        {
            "param_sets": param_sets,
            "anchors": anchors,
            "device": device,
            "training_dataloader": training_dataloader,
            "validation_dataloader": validation_dataloader,
            "testing_dataloader": testing_dataloader,
            "accumulate_steps": params["effective_batch_size"] // params["batch_size"],
        }
    )

    parser = argparse.ArgumentParser(
        "Script to train, test or summarize results from previous training"
    )

    parser.add_argument(
        "--mode",
        help="Select 'train', 'test' or 'summary'.",
        type=str,
        choices=["train", "test", "summary"],
        required=True,
    )
    parser.add_argument(
        "--model-folder",
        help="Folder name where models are stored in **/models/** for 'test' and 'summary'.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model-name", help="Name of model to use for 'test'.", type=str, default=""
    )
    args = parser.parse_args()
    mode = args.mode
    model_folder = args.model_folder
    model_name = args.model_name

    if mode == "test":
        if model_folder == "" and model_name == "":
            raise ValueError(
                "model_folder and model_name must be passed for mode 'test'."
            )
        if model_folder == "":
            raise ValueError("model_folder must be passed for mode 'test'.")
        if model_name == "":
            raise ValueError("model_name must be passed for mode 'test'.")

    if mode == "summary":
        if model_folder == "":
            raise ValueError("model_folder must be passed for mode 'summary'.")

    if mode == "train":
        train_with_hyperparameter_grid_search(**params)

    model_dir = os.path.join(
        "models",
        model_folder,
    )
    if mode == "test":
        model_path = os.path.join(
            model_dir,
            model_name,
        )

        test_model(
            model_path=model_path,
            data_loader=testing_dataloader,
            eval_iou_threshold_NMS=0.3,
            eval_iou_threshold_mAP=[0.5, 0.95, 0.05],
            vis_conf_threshold=0.99,
            vis_iou_threshold_NMS=0.1,
            **params,
        )

    if mode == "summary":
        csv_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            model_dir,
            "data.csv",
        )
        plot_hyperparam_search_summary(csv_path)
