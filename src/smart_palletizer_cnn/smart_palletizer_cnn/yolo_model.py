from typing import List

import numpy as np
import torch
from torch import nn
from torchvision.ops import batched_nms


class Conv2DBlock(nn.Module):
    """A block containing 2D conv, batch nomalization and ReLU layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int = 2,
    ) -> None:
        """Initializes the block.

        Args:
            in_channels (int): Number of input channels to the Conv2D layer.
            out_channels (int): Number of output channels from the Conv2D layer.
            kernel_size (int): Size of the kernel for the Conv2D layer.
            padding (int): Padding for the Conv2D layer.
            stride (int, optional): Stride for the Conv2D layer. Defaults to 2.
        """
        super().__init__()

        self.layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Prediction.
        """
        return self.layers(x)


class YOLOBackbone(nn.Module):
    """YOLO model's backbone."""

    def __init__(
        self,
        num_downsampling_conv_blocks: int,
        num_nondownsampling_conv_blocks: int,
        in_channels: int,
        first_layer_out_channels: int,
        kernel_size: int,
    ) -> None:
        """Initializes the backbone.

        Args:
            num_downsampling_conv_blocks (int): Number of Conv2D blocks that downsample the input.
            num_nondownsampling_conv_blocks (int): Number of Conv2D blocks that don't downsample the input.
            in_channels (int): Number of channels in the input image.
            first_layer_out_channels (int): Output channels of the first Conv2D block.
            kernel_size (int): Size of the kernel.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the backbone.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Prediction.
        """
        return self.layers(x)


class YOLOHead(nn.Module):
    """Head of the YOLO network."""

    def __init__(self, input_filters: int, num_classes: int, num_anchors: int) -> None:
        """Initializes the head.

        Args:
            input_filters (int): Number of input channels into the head.
            num_classes (int): Number of classes in the prediction.
            num_anchors (int): Number of anchors used to predict.
        """
        super().__init__()
        self.input_filters = input_filters
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(
            input_filters, num_anchors * (5 + num_classes), kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the head.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Prediction.
        """
        B, _, S, _ = x.shape

        x = self.detector(x)  # (B, A*(5+C), S, S)

        x = x.view(B, self.num_anchors, 5 + self.num_classes, S, S)

        x = x.permute(0, 3, 4, 1, 2).contiguous()

        return x


class YOLOModel(nn.Module):
    """Assembled YOLO model."""

    def __init__(
        self,
        backbone_num_downsampling_conv_blocks: int,
        backbone_num_nondownsampling_conv_blocks: int,
        backbone_in_channels: int,
        backbone_first_layer_out_channels: int,
        backbone_kernel_size: int,
        prediction_head_num_anchors: int,
        prediction_head_num_classes: int,
    ) -> None:
        """Initializes the YOLO model.

        Args:
            backbone_num_downsampling_conv_blocks (int): Number of Conv2D blocks that downsample the input.
            backbone_num_nondownsampling_conv_blocks (int): Number of Conv2D blocks that dont' downsample the input.
            backbone_in_channels (int): Number of channels in the input image.
            backbone_first_layer_out_channels (int): Number of output channels from the first Conv2D block in the backbone.
            backbone_kernel_size (int): Kernel size in the Conv2D block in the backbone.
            prediction_head_num_anchors (int): Number of anchors used to predict in the head.
            prediction_head_num_classes (int): Number of classes in the prediction.
        """
        super().__init__()

        self.backbone_num_downsampling_conv_blocks = backbone_num_downsampling_conv_blocks

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the YOLO network.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Prediction.
        """
        pred = self.backbone(x)
        pred = self.prediction_head(pred)

        return pred
    
    @torch.inference_mode()
    def pred(
        self,
        input: torch.Tensor,
        anchors: torch.Tensor,
        conf_threshold: float,
        iou_threshold_nms: float,
    ) -> List[torch.Tensor]:
        """Run inference on input images and decode predictions.

        Args:
            input (torch.Tensor): Batched input images of shape (B, C, H, W) or a single image (C, H, W).
            anchors (torch.Tensor): Anchors tensor of shape (A, 2) (w, h) normalized to image size.
            conf_threshold (float): Confidence threshold for filtering.
            iou_threshold_nms (float): IoU threshold for NMS.

        Returns:
            List[torch.Tensor]: Decoded predictions (one tensor per batch) on CPU.
        """
        self.eval()

        # Accept single image (C,H,W) or batch (B,C,H,W)
        if input.dim() == 3:
            input = input.unsqueeze(0)

        # Move input to model device
        param = next(self.parameters())
        device = param.device
        input = input.to(device)

        # Forward pass
        preds = self(input)  # (B, S, S, A, 5+C)

        # Compute grid_size from spatial dimension (H)
        grid_size = input.shape[2] // (2 ** self.backbone_num_downsampling_conv_blocks)

        # Ensure anchors on same device
        anchors = anchors.to(preds.device)

        # Decode predictions -> list of tensors on device
        decoded = self.decode_YOLO_encodings(
            preds, grid_size, anchors, conf_threshold, iou_threshold_nms
        )

        # Move decoded tensors to CPU and detach
        decoded_cpu = [d.cpu().detach() for d in decoded]
        return decoded_cpu

    @staticmethod
    def decode_YOLO_encodings(
        pred: torch.Tensor,
        grid_size: int,
        anchors: torch.Tensor,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> List[torch.Tensor]:
        """Decode prediction from the YOLO encodings to a compressed format.

        Args:
            pred (torch.Tensor): Prediction.
            grid_size (int): Size of the grid at which predictions are made.
            anchors (torch.Tensor): Tensor of anchors of various sizes and ratios.
            confidence_threshold (float): Minimum confidence threshold for generating outputs.
            iou_threshold (float): IOU threshold for non maximum suppression.

        Returns:
            List[torch.Tensor]: List of tensors containing the decoded predictions.
        """
        with torch.inference_mode():
            B = pred.shape[0]
            num_classes = pred.shape[-1] - 5

            pred[..., 4] = torch.sigmoid(pred[..., 4])
            pred[..., 5:] = torch.softmax(pred[..., 5:], dim=-1)

            mask = pred[..., 4] > confidence_threshold
            img, h, w, a = mask.nonzero(as_tuple=True)

            if img.numel() == 0:
                return [
                    torch.empty((0, 5 + num_classes), device=pred.device)
                    for _ in range(B)
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
