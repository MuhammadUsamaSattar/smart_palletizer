import math
import os

import matplotlib.pyplot as plt
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
        max_pooling,
        stride=2,
    ):
        super().__init__()
        if max_pooling:
            stride = 1

        self.layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]

        if max_pooling:
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class YOLOBackbone(nn.Module):
    def __init__(
        self,
        num_conv_blocks,
        in_channels,
        first_layer_out_channels,
        kernel_size,
        max_pooling,
    ):
        super().__init__()

        self.layers = []
        curr_in_channels = in_channels
        curr_out_channels = first_layer_out_channels
        for _ in range(1, num_conv_blocks + 1):
            self.layers.append(
                Conv2DBlock(
                    in_channels=curr_in_channels,
                    out_channels=curr_out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    max_pooling=max_pooling,
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
        backbone_num_conv_blocks,
        backbone_in_channels,
        backbone_first_layer_out_channels,
        backbone_kernel_size,
        backbone_max_pooling,
        prediction_head_num_anchors,
        prediction_head_num_classes,
    ):
        super().__init__()

        self.backbone = YOLOBackbone(
            num_conv_blocks=backbone_num_conv_blocks,
            in_channels=backbone_in_channels,
            first_layer_out_channels=backbone_first_layer_out_channels,
            kernel_size=backbone_kernel_size,
            max_pooling=backbone_max_pooling,
        )
        self.prediction_head = YOLOHead(
            backbone_first_layer_out_channels * (2 ** (backbone_num_conv_blocks - 1)),
            prediction_head_num_classes,
            prediction_head_num_anchors,
        )

        # self.generate_anchors(scales, ratios)

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


def visualize_prediction(pred, image, grid_size, anchors, iou_threshold):
    with torch.no_grad():
        image = image.cpu().detach()
        decoded_pred = decode_YOLO_encoding(pred, grid_size, anchors, iou_threshold)
        decoded_pred[:, :2] = decoded_pred[:, :2] - (decoded_pred[:, 2:4] / 2)

        labels = {"bboxes": decoded_pred.cpu().detach()}
        utils.visualize_box_and_pose_data(image, labels, options={"color", "bboxes"})


def decode_YOLO_encoding(pred, grid_size, anchors, iou_threshold):
    mask = torch.sigmoid(pred[..., 4]) > 0.7
    mask = torch.argwhere(mask)
    h, w, a = mask.T

    decoded_pred = torch.stack(
        [
            (w + torch.sigmoid(pred[h, w, a, 0])) / grid_size,
            (h + torch.sigmoid(pred[h, w, a, 1])) / grid_size,
            anchors[a, 0] * torch.exp(pred[h, w, a, 2]),
            anchors[a, 1] * torch.exp(pred[h, w, a, 3]),
            torch.sigmoid(pred[h, w, a, 4]),
        ],
        dim=1,
    )
    decoded_pred = non_max_suppression(decoded_pred, iou_threshold=iou_threshold)

    return decoded_pred


def non_max_suppression(predictions, iou_threshold):
    """Applies NMS to suppress overlapping boxes."""
    if len(predictions) == 0:
        return torch.tensor([])
    # print(len(predictions))

    # predictions[:, 2:4] = predictions[:, 0:2] + predictions[:, 2:4]
    # predictions = list(predictions)
    # predictions = sorted(predictions, key=lambda x: x[4], reverse=False)
    _, idx = torch.sort(predictions[:, 4], dim=0, descending=True)
    predictions = predictions[idx]
    final_predictions = []

    while len(predictions) != 0:
        final_predictions.append(predictions[0, :4])
        ious = box_iou(predictions[0, :4], predictions[1:, :4], fmt="cxcywh")
        mask = ious < iou_threshold
        predictions = predictions[1:][mask[0]]
        # predictions = [p for p in predictions if compute_iou(best[:4], p[:4]) < iou_threshold]

    return torch.stack(final_predictions, dim=0)


# def compute_iou(box1, box2):
#     """Computes IoU between two bounding boxes."""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     intersection = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = box1_area + box2_area - intersection

#     return intersection / union


def train(
    training_dataloader,
    backbone_num_conv_blocks,
    backbone_in_channels,
    backbone_first_layer_out_channels,
    backbone_kernel_size,
    backbone_max_pooling,
    prediction_head_num_classes,
    image_size,
    anchors,
    lambda_coords,
    lambda_no_obj,
    num_epochs,
    device,
    lr=0.001,
    print_loss_after_batches=False,
    visualize_after_batches=False,
    visualize_after_epoch=False,
    visualize_after_training=False,
    iou_threshold=0.25,
):

    grid_size = image_size // (2**backbone_num_conv_blocks)
    model = YOLOModel(
        backbone_num_conv_blocks=backbone_num_conv_blocks,
        backbone_in_channels=backbone_in_channels,
        backbone_first_layer_out_channels=backbone_first_layer_out_channels,
        backbone_kernel_size=backbone_kernel_size,
        backbone_max_pooling=backbone_max_pooling,
        prediction_head_num_anchors=anchors.shape[0],
        prediction_head_num_classes=prediction_head_num_classes,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_fn(
        lambda_coords=lambda_coords,
        lambda_no_obj=lambda_no_obj,
        grid_size=grid_size,
        anchors=anchors,
        num_classes=prediction_head_num_classes,
        device=device,
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for i, (X, y) in enumerate(training_dataloader):
            X, y = X.to(device), y

            pred = model(X)

            loss = loss_fn(pred=pred, labels=y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % max(len(training_dataloader) // 10, 1) == 0:
                if print_loss_after_batches:
                    print(f"Batch {i+1}/{len(training_dataloader)}, {loss.item()=}")
                if visualize_after_batches:
                    visualize_prediction(
                        pred[0], X[0], grid_size, anchors, iou_threshold=iou_threshold
                    )

        print(
            f"Epoch {epoch+1}/{num_epochs}, Average epoch loss = {epoch_loss/len(training_dataloader)}"
        )

        if visualize_after_epoch:
            visualize_prediction(
                pred[0], X[0], grid_size, anchors, iou_threshold=iou_threshold
            )

    os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models"), exist_ok=True)
    model_path  = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "models",
            "_".join(
                [
                    model._get_name(),
                    f"lr-{lr}",
                    f"lambda-coords-{lambda_coords}",
                    f"lambda-no-obj-{lambda_no_obj}",
                    f"backbone-num-conv-blocks-{backbone_num_conv_blocks}",
                    f"backbone-first-layer-out-channels-{backbone_first_layer_out_channels}",
                    f"backbone-kernel-size-{backbone_kernel_size}",
                    f"backbone-max-pooling-{backbone_max_pooling}",
                ]
            ),
        )
    torch.save(
        model.state_dict(),
        model_path
    )
    print(f"Model saved to {model_path}")

    if visualize_after_training:
        visualize_prediction(
            pred[0], X[0], grid_size, anchors, iou_threshold=iou_threshold
        )


if __name__ == "__main__":
    params = {
        "backbone_in_channels": 2,
        "prediction_head_num_classes": 2,
        "image_size": 512,
        "num_epochs": 1,
        "batch_size": 82,
        "print_loss_after_batches": True,
        "visualize_after_batches": False,
        "visualize_after_epoch": False,
        "visualize_after_training": True,
        "iou_threshold": 0.25,
    }
    torch.set_printoptions(threshold=10_000)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    anchors = generate_anchors(
        [0.05, 0.1, 0.2, 0.3], [1 / 1.65, 1 / 1.35, 1.35, 1.65]
    ).to(device)

    params.update({
        "backbone_num_conv_blocks": 5,
        "backbone_first_layer_out_channels": 16,
        "backbone_kernel_size": 3,
        "backbone_max_pooling": True,
        "lambda_coords": 50,
        "lambda_no_obj": 0.5,
        "lr": 0.001,
    })

    data = BoxAndPoseDataset(
        "data/synthetic_data",
        "coco_annotations.json",
        "hdf5",
        transform_feature=True,
        transform_label=True,
        image_size=params["image_size"],
        dtype=torch.float32,
    )

    training_data, testing_data = random_split(data, [0.01, 0.99])

    training_dataloader = DataLoader(
        training_data,
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

    # train(
    #     training_dataloader,
    #     backbone_num_conv_blocks=params["backbone_num_conv_blocks"],
    #     backbone_in_channels=params["backbone_in_channels"],
    #     backbone_first_layer_out_channels=params["backbone_first_layer_out_channels"],
    #     backbone_kernel_size=params["backbone_kernel_size"],
    #     backbone_max_pooling=params["backbone_max_pooling"],
    #     prediction_head_num_classes=params["prediction_head_num_classes"],
    #     image_size=params["image_size"],
    #     anchors=anchors,
    #     lambda_coords=params["lambda_coords"],
    #     lambda_no_obj=params["lambda_no_obj"],
    #     num_epochs=params["num_epochs"],
    #     device=device,
    #     lr=params["lr"],
    #     print_loss_after_batches=params["print_loss_after_batches"],
    #     visualize_after_batches=params["visualize_after_batches"],
    #     visualize_after_epoch=params["visualize_after_epoch"],
    #     visualize_after_training=params["visualize_after_training"],
    #     iou_threshold=params["iou_threshold"],
    # )

import torch
from torch.utils.data import DataLoader

# ---------- Fast WH collection ----------
def collect_wh_fast(dataset, batch_size=128, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    wh = []

    for batch in loader:
        for _, label in batch:
            b = label["bboxes"][:, 2:4]  # (N, 2)
            if len(b) > 0:
                wh.append(b)

    return torch.cat(wh, dim=0).to(device)  # (M, 2)


# ---------- Fast IoU ----------
def wh_iou(boxes, anchors):
    # boxes: (M, 2), anchors: (A, 2)
    inter = torch.min(boxes[:, None, :], anchors[None, :, :]).prod(dim=2)
    union = boxes.prod(dim=1, keepdim=True) + anchors.prod(dim=1) - inter
    return inter / union


# ---------- GPU KMeans ----------
def kmeans_anchors_fast(wh, k=4, iters=25):
    idx = torch.randperm(len(wh))[:k]
    anchors = wh[idx].clone()

    for _ in range(iters):
        ious = wh_iou(wh, anchors)      # (M, k)
        best = torch.argmax(ious, dim=1)

        new_anchors = []
        for i in range(k):
            mask = best == i
            new_anchors.append(
                wh[mask].mean(dim=0) if mask.any() else anchors[i]
            )

        anchors = torch.stack(new_anchors)

    return anchors


# ---------- Convert to scales & ratios ----------
def anchors_to_scales_ratios(anchors):
    w, h = anchors[:, 0], anchors[:, 1]
    scales = torch.sqrt(w * h)
    ratios = w / h
    return scales, ratios


# ---------- Main ----------
def generate_anchors_scales_ratios_fast(dataset, k=4, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wh = collect_wh_fast(dataset, batch_size=batch_size, device=device)
    anchors = kmeans_anchors_fast(wh, k=k)
    scales, ratios = anchors_to_scales_ratios(anchors)

    return anchors.cpu(), scales.cpu(), ratios.cpu()


# ---------- Example ----------
anchors, scales, ratios = generate_anchors_scales_ratios_fast(data, k=4)
print("Anchors:\n", anchors)
print("Scales:\n", scales)
print("Ratios:\n", ratios)

