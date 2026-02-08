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
from torch.amp import autocast, GradScaler
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.ops import box_iou, nms, batched_nms


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


def decode_YOLO_encodings(pred, grid_size, anchors, confidence_threshold, iou_threshold):
    with torch.inference_mode():
        B = pred.shape[0]
        num_classes = pred.shape[-1] - 5

        pred[..., 4] = torch.sigmoid(pred[..., 4])
        pred[..., 5:] = torch.softmax(pred[..., 5:], dim=-1)

        mask = pred[..., 4] > confidence_threshold
        img, h, w, a = mask.nonzero(as_tuple=True)

        if img.numel() == 0:
            return [
                torch.empty((0, 5 + num_classes), device=pred.device) for _ in range(B)
            ]

        boxes = pred[img, h, w, a]

        xy = torch.stack(
            [
                (w + torch.sigmoid(boxes[:, 0])) / grid_size,
                (h + torch.sigmoid(boxes[:, 1])) / grid_size,
            ],
            dim=1,
        )

        wh = anchors[a] * torch.exp(boxes[:, 2:4])

        xy1 = xy - wh / 2
        xy2 = xy + wh / 2
        boxes_xyxy = torch.cat([xy1, xy2], dim=1)

        obj = boxes[:, 4]
        cls_probs = boxes[:, 5:]

        cls_ids = cls_probs.argmax(dim=1)
        scores = obj * cls_probs.max(dim=1).values

        group_ids = img * num_classes + cls_ids

        keep = batched_nms(
            boxes_xyxy,
            scores,
            group_ids,
            iou_threshold,
        )

        final = torch.cat(
            [
                boxes_xyxy[keep],
                scores[keep, None],
                cls_probs[keep],
            ],
            dim=1,
        )

        final[:, 2:4] = final[:, 2:4] - final[:, :2]
        decoded_pred = [final[img[keep] == i] for i in range(B)]

        return decoded_pred


def evaluate_model(
    model,
    data_loader,
    grid_size,
    iou_threshold_NMS,
    iou_threshold_mAP,
    desc="Validation",
):
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
                decode_YOLO_encodings(
                pred, grid_size, anchors, 0.0, iou_threshold_NMS
                )
            )
            labels.extend(y)
        mAP = compute_mAP(decoded_preds, labels, iou_thresholds, device)

        return mAP


def compute_mAP(preds, labels, iou_thresholds, device):
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
    accumulate_steps,
    use_amp,
    lr: list = [0.001],
    print_info_after_batches=False,
    print_info_after_epoch=False,
    visualize_after_batches=False,
    visualize_after_epoch=False,
    visualize_after_training=False,
    iou_threshold_NMS=0.30,
    iou_threshold_mAP=0.50,
    drop_models_after_epochs=5,
    models_to_drop=2 / 3,
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


def drop_runs(models_to_drop, n_remaining_models, runs, n_active_models, models_dir):
    prev_n_active_models = n_active_models
    n_active_models = max(
        math.floor(n_active_models * (1-models_to_drop)), n_remaining_models
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
    accumulate_steps,
    use_amp,
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
    accumulate_steps,
    use_amp,
):
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

        if (i+1) % accumulate_steps == 0 or (i+1) == len(training_dataloader):
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
                bins = np.logspace(
                    np.log10(subset[p].min()), np.log10(subset[p].max()), 8
                )
                subset[p] = pd.cut(subset[p], bins)

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
        #     f"mAP@{eval_iou_threshold_mAP}={evaluate_model(model, data_loader, image_size // (2 ** int(params["bb-ndscb"])), eval_iou_threshold_NMS, eval_iou_threshold_mAP, desc="Test")}"
        # )
        for X, y in data_loader:
            X = X.to(device)
            preds = model(X)
            preds = decode_YOLO_encodings(
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
        "iou_threshold_mAP": [0.5, 0.95, 0.05],
        "lambda_coords": [8, 10, 12],
        "lambda_no_obj": [0.0625, 0.125, 0.250, 0.375],
        "lr": [0.000320, 0.000160, 0.000480, 0.000240, 0.000800],
        "drop_models_after_epochs": 5,
        "models_to_drop": 1 / 6,
        "n_remaining_models": 3,
        "use_amp": True,
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
        # [0.98, 0.01, 0.01],
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

    param_sets = [
        [3, 10, 32, 5, 12, 0.375, 0.00048],
        [3, 10, 32, 5, 10, 0.375, 0.00048],
        [3, 10, 32, 5, 10, 0.375, 0.00016],
        [3, 10, 32, 5, 10, 0.250, 0.00048],
        [3, 7, 16, 3, 8, 0.375, 0.00048],
        [3, 7, 16, 3, 10, 0.375, 0.00080],
    ]
    # arch_param_sets = [
    #     [3, 10, 32, 5],
    #     [3, 7, 16, 3],
    # ]

    # # loss / optimization space
    # loss_space = list(itertools.product(
    #     params["lambda_coords"],
    #     params["lambda_no_obj"],
    #     params["lr"],
    # ))

    # param_sets = []

    # for arch in arch_param_sets:
    #     # sample up to some number of unique loss configs per architecture
    #     sampled_loss_params = random.sample(
    #         loss_space,
    #         k=min(15, len(loss_space))
    #     )
    #     for lambda_coords, lambda_no_obj, lr in sampled_loss_params:
    #         # lr = 10 ** random.uniform(-3.5, -2.5)
    #         param_sets.append([
    #             arch[0],  # bb-ndscb
    #             arch[1],  # bb-nndscb
    #             arch[2],  # bb-flc
    #             arch[3],  # bb-ks
    #             lambda_coords,
    #             lambda_no_obj,
    #             lr,
    #         ])

    # random.shuffle(param_sets)

    params.update(
        {
            "param_sets": param_sets,
            "anchors": anchors,
            "device": device,
            "training_dataloader": training_dataloader,
            "validation_dataloader": validation_dataloader,
            "testing_dataloader": testing_dataloader,
            "accumulate_steps": params["effective_batch_size"] // params["batch_size"]
        }
    )

    modes = ["train", "test", "summary"]
    mode = modes[1]

    if mode == "train":
        train_with_hyperparameter_grid_search(**params)

    model_dir = os.path.join(
        "models",
        "final_training",
    )
    if mode == "test":
        model_name = "YOLOModel_lr-0.00048_lmbd-co-10_lmbd-no-0.25_bb-ndscb-3_bb-nndscb-10_bb-flc-32_bb-ks-5"
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
