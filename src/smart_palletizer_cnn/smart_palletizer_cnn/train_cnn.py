from collections import defaultdict
import csv
from datetime import datetime
import gc
import glob
import itertools
import math
import os
import random
from tqdm import tqdm

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from smart_palletizer_cnn.box_and_pose_dataset import BoxAndPoseDataset
from smart_palletizer_utils import utils
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.ops import box_iou


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=2,
    ):
        super().__init__()

        self.layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class YOLOBackbone(nn.Module):
    def __init__(
        self,
        num_downsampling_conv_blocks,
        num_nondownsampling_conv_blocks,
        in_channels,
        first_layer_out_channels,
        kernel_size,
    ):
        super().__init__()

        self.layers = []
        curr_in_channels = in_channels
        curr_out_channels = first_layer_out_channels

        weights = np.arange(1, num_downsampling_conv_blocks + 1) ** 3
        weights = weights / weights.sum()
        raw_counts = weights * num_nondownsampling_conv_blocks
        num_refinement_blocks = np.floor(raw_counts).astype(int)

        # Distribute remainder
        remainder = num_nondownsampling_conv_blocks - num_refinement_blocks.sum()
        if remainder != 0:
            for i in np.argsort(raw_counts - num_refinement_blocks)[-remainder:]:
                num_refinement_blocks[i] += 1

        for i in range(num_downsampling_conv_blocks):
            self.layers.append(
                Conv2DBlock(
                    in_channels=curr_in_channels,
                    out_channels=curr_out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=2,
                )
            )

            for j in range(num_refinement_blocks[i]):
                self.layers.append(
                    Conv2DBlock(
                        in_channels=curr_out_channels,
                        out_channels=curr_out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=1,
                    )
                )
            curr_in_channels, curr_out_channels = (
                curr_out_channels,
                curr_out_channels * 2,
            )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class YOLOHead(nn.Module):
    def __init__(self, input_filters, num_classes, num_anchors):
        super().__init__()
        self.input_filters = input_filters
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(
            input_filters, num_anchors * (5 + num_classes), kernel_size=1
        )

    def forward(self, x):
        """
        x: (B, C, S, S)
        return: (B, S, S, A, 5+C)
        """
        B, _, S, _ = x.shape

        x = self.detector(x)  # (B, A*(5+C), S, S)

        x = x.view(B, self.num_anchors, 5 + self.num_classes, S, S)

        x = x.permute(0, 3, 4, 1, 2).contiguous()

        return x


class YOLOModel(nn.Module):
    def __init__(
        self,
        backbone_num_downsampling_conv_blocks,
        backbone_num_nondownsampling_conv_blocks,
        backbone_in_channels,
        backbone_first_layer_out_channels,
        backbone_kernel_size,
        prediction_head_num_anchors,
        prediction_head_num_classes,
    ):
        super().__init__()

        self.backbone = YOLOBackbone(
            num_downsampling_conv_blocks=backbone_num_downsampling_conv_blocks,
            num_nondownsampling_conv_blocks=backbone_num_nondownsampling_conv_blocks,
            in_channels=backbone_in_channels,
            first_layer_out_channels=backbone_first_layer_out_channels,
            kernel_size=backbone_kernel_size,
        )
        self.prediction_head = YOLOHead(
            backbone_first_layer_out_channels
            * (2 ** (backbone_num_downsampling_conv_blocks - 1)),
            prediction_head_num_classes,
            prediction_head_num_anchors,
        )

    def forward(self, x):
        pred = self.backbone(x)
        pred = self.prediction_head(pred)

        return pred


def collate_fn(batch):
    X_batch = torch.stack([item[0] for item in batch], dim=0)
    y_batch = [item[1] for item in batch]  # keep labels as list of dicts
    return X_batch, y_batch


def generate_anchors(scales, ratios):
    """Generates anchor boxes for given scales and aspect ratios."""
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * math.sqrt(ratio)
            height = scale / math.sqrt(ratio)
            anchors.append((width, height))
    return torch.tensor(anchors)


def get_loss_fn(lambda_coords, lambda_no_obj, grid_size, anchors, num_classes, device):
    def get_loss_val(pred, labels):
        labels = encode_labels_to_YOLO_format(
            labels, grid_size, anchors, num_classes, device
        )

        obj_mask = labels[..., 4] == 1.0
        no_obj_mask = labels[..., 4] == 0.0

        num_objs = obj_mask.sum()
        num_no_objs = no_obj_mask.sum()

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


def encode_labels_to_YOLO_format(labels, grid_size, anchors, num_classes, device):
    """
    labels: list of dicts
      - bboxes: (N, 4) normalized (cx, cy, w, h)
      - categories: (N,) class indices
    anchors: (A, 2) normalized (w, h)
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


def visualize_prediction(pred, image):
    image = image.cpu().detach()

    labels = {"bboxes": pred.cpu().detach()}
    utils.visualize_box_and_pose_data(image, labels, options={"color", "bboxes"})
    del labels


def decode_YOLO_encoding(pred, grid_size, anchors, confidence_threshold, iou_threshold):
    decoded_pred = [[] for _ in range(pred.shape[0])]
    pred = pred.clone()
    pred[..., 4] = torch.sigmoid(pred[..., 4])
    pred[..., 5:] = torch.softmax(pred[..., 5:], dim=4)

    for b in range(pred.shape[0]):
        mask = torch.argwhere(pred[b, ..., 4] > confidence_threshold)
        h, w, a = mask.T

        decoded_pred[b] = torch.stack(
            [
                (w + torch.sigmoid(pred[b, h, w, a, 0])) / grid_size,
                (h + torch.sigmoid(pred[b, h, w, a, 1])) / grid_size,
                anchors[a, 0] * torch.exp(pred[b, h, w, a, 2]),
                anchors[a, 1] * torch.exp(pred[b, h, w, a, 3]),
                pred[b, h, w, a, 4],
                pred[b, h, w, a, 5],
                pred[b, h, w, a, 6],
            ],
            dim=1,
        )
        decoded_pred[b][:, :2] = decoded_pred[b][:, :2] - (decoded_pred[b][:, 2:4] / 2)
        decoded_pred[b] = non_max_suppression(
            decoded_pred[b], iou_threshold=iou_threshold
        )

    return decoded_pred


def non_max_suppression(predictions, iou_threshold):
    """
    Applies class-wise Non-Max Suppression.
    predictions: (N, 5 + C) tensor
    """
    if predictions.numel() == 0:
        return torch.empty((0, predictions.shape[1]), device=predictions.device)

    num_classes = predictions.shape[1] - 5
    final_predictions = []

    for cls in range(num_classes):
        class_scores = predictions[:, 4] * predictions[:, 5 + cls]
        mask = class_scores > 0

        if mask.sum() == 0:
            continue

        cls_preds = predictions[mask].clone()
        cls_scores = class_scores[mask]

        _, idx = torch.sort(cls_scores, descending=True)
        cls_preds = cls_preds[idx]

        while len(cls_preds) > 0:
            best = cls_preds[0]
            final_predictions.append(best)

            if len(cls_preds) == 1:
                break

            ious = box_iou(
                best[:4].unsqueeze(0), cls_preds[1:, :4], fmt="xywh"
            ).squeeze(0)

            cls_preds = cls_preds[1:][ious < iou_threshold]

    return (
        torch.stack(final_predictions, dim=0)
        if final_predictions
        else torch.empty((0, predictions.shape[1]), device=predictions.device)
    )


def evaluate_model(
    model,
    data_loader,
    grid_size,
    iou_threshold_NMS,
    iou_threshold_mAP,
    desc="Validation",
):
    model.eval()
    with torch.inference_mode():
        decoded_preds = []
        labels = []
        for X, y in tqdm(
            data_loader, desc=desc + " Batches", unit="batch", leave=False
        ):
            X = X.to(device)
            pred = model(X)
            decoded_preds.extend(
                decode_YOLO_encoding(pred, grid_size, anchors, 0.0, iou_threshold_NMS)
            )
            labels.extend(y)
        mAP = compute_mAP(decoded_preds, labels, iou_threshold_mAP, device)

    return mAP


def compute_mAP(preds, labels, iou_threshold, device):
    num_classes = preds[0].shape[1] - 5
    average_precisions = []

    for cls in range(num_classes):
        confs = []
        true_positives = []
        total_ground_truths = 0

        for lbl in labels:
            total_ground_truths += (lbl["categories"][:, cls] == 1).sum().item()

        if total_ground_truths == 0:
            average_precisions.append(0.0)
            continue

        for img_preds, img_labels in zip(preds, labels):
            scores = img_preds[:, 4] * img_preds[:, 5 + cls]
            keep = img_preds[:, 5 + cls] > 0.5

            img_preds = img_preds[keep]
            scores = scores[keep]

            ground_truth_boxes = img_labels["bboxes"][
                img_labels["categories"][:, cls] == 1
            ].to(device)

            if len(img_preds) == 0:
                continue

            confs.append(scores)

            if len(ground_truth_boxes) == 0:
                true_positives.append(torch.zeros(len(img_preds), device=device))
                continue

            ious = box_iou(img_preds[:, :4], ground_truth_boxes, fmt="xywh")
            max_ious, ground_truth_idx = ious.max(dim=1)

            assigned = torch.zeros(
                len(ground_truth_boxes), dtype=torch.bool, device=device
            )
            true_positive = torch.zeros(len(img_preds), device=device)

            for i in torch.argsort(scores, descending=True):
                if max_ious[i] >= iou_threshold and not assigned[ground_truth_idx[i]]:
                    true_positive[i] = 1
                    assigned[ground_truth_idx[i]] = True

            true_positives.append(true_positive)

        if len(confs) == 0:
            average_precisions.append(0.0)
            continue

        confs = torch.cat(confs)
        true_positives = torch.cat(true_positives)

        order = torch.argsort(confs, descending=True)
        true_positives = true_positives[order]

        false_positives = 1 - true_positives
        true_positives_cum = torch.cumsum(true_positives, 0)
        false_positives_cum = torch.cumsum(false_positives, 0)

        recalls = true_positives_cum / total_ground_truths
        precisions = true_positives_cum / (
            true_positives_cum + false_positives_cum + 1e-9
        )

        # Precision envelope
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = torch.maximum(precisions[i], precisions[i + 1])

        # Remove duplicate precision points
        keep = torch.ones_like(precisions, dtype=torch.bool)
        keep[1:] = precisions[1:] != precisions[:-1]

        precisions = precisions[keep]
        recalls = recalls[keep]

        # Area under PR curve
        AP = torch.sum((recalls[1:] - recalls[:-1]) * precisions[:-1])
        average_precisions.append(AP.item())

    return sum(average_precisions) / len(average_precisions)


def train_with_hyperparameter_grid_search(
    param_sets,
    training_dataloader,
    validation_dataloader,
    testing_dataloader,
    backbone_num_downsampling_conv_blocks: list,
    backbone_num_nondownsampling_conv_blocks: list,
    backbone_in_channels,
    backbone_first_layer_out_channels: list,
    backbone_kernel_size: list,
    prediction_head_num_classes,
    image_size,
    anchors,
    lambda_coords: list,
    lambda_no_obj: list,
    num_epochs,
    device,
    lr: list = [0.001],
    print_info_after_batches=False,
    print_info_after_epoch=False,
    visualize_after_batches=False,
    visualize_after_epoch=False,
    visualize_after_training=False,
    iou_threshold_NMS=0.30,
    iou_threshold_mAP=0.50,
    drop_models_after_epochs=5,
    models_to_keep=1 / 3,
    n_remaining_models=5,
    max_patience=5,
    min_delta=0.001,
    verbose=False,
    **kwargs,
):

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
                        f"run_id={run["run_id"]}, model_name={run["model_name"]}, mAP={run["mAP"]:.5f}, Active"
                    )
                else:
                    tqdm.write(
                        f"run_id={run["run_id"]}, model_name={run["model_name"]}, mAP={run["mAP"]:.5f}, Early Stopped"
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
                tqdm.write(f"Dropping {1-models_to_keep:.0%} of worst active models!")
                n_active_models = drop_runs(
                    models_to_keep,
                    n_remaining_models,
                    runs,
                    n_active_models,
                    models_dir,
                )

    for filename in glob.glob(models_dir + "/curr_*"):
        os.remove(filename)

    print("################################")
    for i in range(n_active_models):
        model = YOLOModel(
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
            f"Test mAP={evaluate_model(
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


def drop_runs(models_to_keep, n_remaining_models, runs, n_active_models, models_dir):
    prev_n_active_models = n_active_models
    n_active_models = max(
        math.floor(n_active_models * models_to_keep), n_remaining_models
    )
    for run in runs[n_active_models:prev_n_active_models]:
        run["active"] = False
        if run["patience"] > 0:
            os.remove(os.path.join(models_dir, "curr_" + run["model_name"]))

    return n_active_models


def train_checkpoint_for_one_epoch(
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
):
    if verbose:
        tqdm.write(f"Processing run_id={run["run_id"]}")

    model = YOLOModel(
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
    backbone_in_channels,
    prediction_head_num_classes,
    image_size,
    anchors,
    device,
    verbose,
    param_sets,
    models_dir,
):
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

        model = YOLOModel(
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


def free_memory(model, optimizer):
    if model is not None:
        model = model.to("cpu")  # move weights off GPU
    if optimizer is not None:
        optimizer.state.clear()  # drop optimizer tensors

    del optimizer
    del model

    gc.collect()
    torch.cuda.empty_cache()


def save_model(
    model,
    models_dir,
    model_name,
    optimizer,
    mAP,
    loss,
    epoch,
    best=True,
    verbose=False,
):
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
    model,
    models_dir,
    model_name,
    optimizer,
    device,
    best=True,
    verbose=False,
):
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
    grid_size,
    model,
    optimizer,
    loss_fn,
    epoch,
):
    model.train()
    epoch_loss = 0

    for i, (X, y) in enumerate(
        tqdm(training_dataloader, desc="Training Batches", unit="batch", leave=False)
    ):
        X, y = X.to(device), y

        pred = model(X)

        loss = loss_fn(pred=pred, labels=y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

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
            f"Epoch {epoch+1}/{num_epochs}, Average epoch loss = {epoch_loss/len(training_dataloader)}, {mAP=}"
        )

    if visualize_after_epoch:
        visualize_prediction(pred[0], X[0])
    return mAP, epoch_loss / len(training_dataloader), X[0], pred[0]


def plot_hyperparam_search_summary(csv_path, top_k=20, interaction_pairs=None):
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
                params[key] = v
        parsed_params.append(params)
        for k, v in params.items():
            hyperparams[k].append(v)
    hyperparams = dict(hyperparams)

    # ==========================================================
    # 2. FIGURE 1: SUMMARY + HYPERPARAM BOX PLOTS (3x4)
    # ==========================================================
    fig1 = plt.figure(figsize=(16, 12))
    gs1 = GridSpec(3, 4, figure=fig1)
    plot_idx = 0

    def next_ax1():
        nonlocal plot_idx
        row, col = divmod(plot_idx, 4)
        ax = fig1.add_subplot(gs1[row, col])
        plot_idx += 1
        return ax

    # --- Top-K leaderboard
    ax = next_ax1()
    top_idx = max_mAP.sort_values(ascending=False).head(top_k).index
    ax.barh(range(len(top_idx)), max_mAP[top_idx][::-1])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([f"run_{i}" for i in top_idx][::-1])
    ax.set_title(f"Top {top_k} Models (Max mAP)")
    ax.set_xlabel("mAP")

    # --- Distribution
    ax = next_ax1()
    ax.hist(max_mAP.dropna(), bins=30)
    ax.set_title("Distribution of Max mAP")

    # --- Survival
    ax = next_ax1()
    ax.scatter(run_ids, survival_len, s=10)
    ax.set_title("Epochs Survived")
    ax.set_xlabel("Run ID")

    # --- Best mAP vs run_id
    ax = next_ax1()
    ax.scatter(run_ids, max_mAP, s=10)
    ax.set_title("Best mAP vs Run ID")
    ax.set_xlabel("Run ID")

    # --- mAP vs epochs for top_k
    ax = next_ax1()
    for i in top_idx:
        y = values.iloc[:, i].dropna().values
        # prepend 0.0 at epoch 0
        y = np.insert(y, 0, 0.0)
        x = np.arange(len(y))  # epochs 0, 1, 2, ...
        ax.plot(x, y, alpha=0.5, label=f"run_{i}")

    ax.set_title(f"Top-{top_k} mAP vs Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])

    # --- Hyperparameter boxplots
    for hp, vals in hyperparams.items():
        grouped = defaultdict(list)
        for i, v in enumerate(vals):
            if not np.isnan(max_mAP.iloc[i]):
                grouped[v].append(max_mAP.iloc[i])
        if len(grouped) < 2:
            continue
        if plot_idx >= 12:
            break
        ax = next_ax1()
        labels = list(grouped.keys())
        data = [grouped[k] for k in labels]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(hp, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    # --- Pareto front
    if plot_idx < 12:
        points = np.column_stack([survival_len.values, max_mAP.values])
        pareto = []
        for i, p in enumerate(points):
            if not any(
                (q[0] <= p[0] and q[1] >= p[1]) and (q != p).any() for q in points
            ):
                pareto.append(p)
        if pareto:
            pareto = np.array(sorted(pareto, key=lambda x: x[0]))
            ax = next_ax1()
            ax.scatter(survival_len, max_mAP, alpha=0.4, label="All models")
            ax.plot(pareto[:, 0], pareto[:, 1], "r-o", label="Pareto front")
            ax.set_xlabel("Epochs Trained")
            ax.set_ylabel("Max mAP")
            ax.set_title("Pareto Front")
            ax.legend()

    # Turn off unused axes
    for i in range(plot_idx, 12):
        row, col = divmod(i, 4)
        fig1.add_subplot(gs1[row, col]).axis("off")

    plt.tight_layout()

    # ==========================================================
    # 3. FIGURE 2: INTERACTION HEATMAPS (3x7)
    # ==========================================================
    if interaction_pairs is None:
        keys = list(hyperparams.keys())
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
        table = defaultdict(list)
        for i, params in enumerate(parsed_params):
            if p1 in params and p2 in params and not np.isnan(max_mAP.iloc[i]):
                table[(params[p1], params[p2])].append(max_mAP.iloc[i])
        if not table:
            continue
        xs = sorted({k[0] for k in table})
        ys = sorted({k[1] for k in table})
        Z = np.full((len(ys), len(xs)), np.nan)
        for (x, y), vals in table.items():
            Z[ys.index(y), xs.index(x)] = np.mean(vals)
        ax = next_ax2()
        im = ax.imshow(Z, origin="lower", aspect="auto")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=30)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys)
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f"{p1} x {p2}")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Max mAP")

    # Turn off unused axes
    for i in range(plot_idx, 21):
        row, col = divmod(i, 7)
        fig2.add_subplot(gs2[row, col]).axis("off")

    plt.tight_layout()

    # ==========================================================
    # 4. SHOW BOTH FIGURES
    # ==========================================================
    plt.show()


def test_model(
    model_path,
    data_loader,
    image_size,
    backbone_in_channels,
    anchors,
    eval_iou_threshold_mAP,
    eval_iou_threshold_NMS,
    vis_iou_threshold_NMS,
    vis_conf_threshold,
    prediction_head_num_classes,
    device,
    **kwargs,
):
    model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        model_dir,
        model_path,
    )
    model_dir, model_name = os.path.split(model_path)

    parts = model_name.split("_")[1:]
    params = {}
    for part in parts:
        *k, v = part.split("-", -1)
        k = "-".join(k)
        params[k] = v

    model = YOLOModel(
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
        # tqdm.write(
        #     f"mAP={evaluate_model(model, data_loader, image_size // (2 ** int(params["bb-ndscb"])), eval_iou_threshold_NMS, eval_iou_threshold_mAP, desc="Test")}"
        # )
        for X, y in data_loader:
            X = X.to(device)
            preds = model(X)
            preds = decode_YOLO_encoding(
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
        "backbone_num_downsampling_conv_blocks": [3, 4],
        "backbone_num_nondownsampling_conv_blocks": [0, 3, 5, 7, 10],
        "backbone_first_layer_out_channels": [16, 32],
        "backbone_kernel_size": [3, 5, 7],
        "prediction_head_num_classes": 2,
        "image_size": 256,
        "num_epochs": 3,
        "batch_size": 32,
        "print_info_after_batches": False,
        "print_info_after_epoch": False,
        "visualize_after_batches": False,
        "visualize_after_epoch": False,
        "visualize_after_training": False,
        "iou_threshold_NMS": 0.30,
        "iou_threshold_mAP": 0.50,
        "lambda_coords": [50],
        "lambda_no_obj": [0.5],
        "lr": [0.001],
        "drop_models_after_epochs": 5,
        "models_to_drop": 1 / 3,
        "n_remaining_models": 5,
    }

    # params = {
    #     "backbone_in_channels": 2,
    #     "backbone_num_downsampling_conv_blocks": [5],
    #     "backbone_num_nondownsampling_conv_blocks": [5],
    #     "backbone_first_layer_out_channels": [16],
    #     "backbone_kernel_size": [3],
    #     "prediction_head_num_classes": 2,
    #     "image_size": 512,
    #     "num_epochs": 100,
    #     "batch_size": 32,
    #     "print_info_after_batches": False,
    #     "print_info_after_epoch": False,
    #     "visualize_after_batches": False,
    #     "visualize_after_epoch": False,
    #     "visualize_after_training": False,
    #     "iou_threshold_NMS": 0.30,
    #     "iou_threshold_mAP": 0.75,
    #     "lambda_coords": [50],
    #     "lambda_no_obj": [0.1],
    #     "lr": [0.001],
    #     "drop_models_after_epochs": 5,
    #     "models_to_drop": 1 / 3,
    #     "n_remaining_models": 5,
    # }
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
        # [0.001, 0.001, 0.998],
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

    param_sets = itertools.product(
        params["backbone_num_downsampling_conv_blocks"],
        params["backbone_num_nondownsampling_conv_blocks"],
        params["backbone_first_layer_out_channels"],
        params["backbone_kernel_size"],
        params["lambda_coords"],
        params["lambda_no_obj"],
        params["lr"],
    )

    param_sets = random.sample(list(param_sets), 15)

    params.update(
        {
            "param_sets": param_sets,
            "anchors": anchors,
            "device": device,
            "training_dataloader": training_dataloader,
            "validation_dataloader": validation_dataloader,
            "testing_dataloader": testing_dataloader,
        }
    )

    modes = ["train", "test", "summary"]
    mode = modes[1]

    if mode == "train":
        train_with_hyperparameter_grid_search(**params)

    model_dir = os.path.join(
            "models",
            "2026-01-13_21-35-01",
        )
    if mode == "test":     
        model_name="YOLOModel_lr-0.01_lmbd-co-10_lmbd-no-0.1_bb-ndscb-5_bb-nndscb-0_bb-flc-16_bb-ks-3"
        model_path = os.path.join(
            model_dir,
            model_name,
        )
        
        test_model(
            model_path=model_path,
            data_loader=testing_dataloader,
            eval_iou_threshold_NMS=0.3,
            eval_iou_threshold_mAP=0.5,
            vis_conf_threshold=0.7,
            vis_iou_threshold_NMS=0.3,
            **params,
        )

    if mode == "summary":
        csv_path = os.path.join(
            model_dir,
            "data.csv",
        )
        plot_hyperparam_search_summary(csv_path)
