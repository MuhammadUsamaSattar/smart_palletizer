from collections import defaultdict
import json
import os
from typing import Tuple
from pycocotools import mask as maskUtils

import h5py
import numpy as np
from smart_palletizer_utils import utils
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import (
    to_dtype,
    rgb_to_grayscale,
    pad,
    resize,
    resize_bounding_boxes,
)


class BoxAndPoseDataset(Dataset):
    """Dataset class for box and pose detection."""

    def __init__(
        self,
        root_path: str,
        coco_annotations_file: str,
        hdf5_dir: str,
        transform_feature: bool = True,
        transform_label: bool = True,
        image_size: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initializes paths, filenames and transformation booleans.

        Args:
            root_path (str): Root of the dataset dir.
            coco_annotations_file (str): Coco annotation filename.
            hdf5_dir (str): Directory containing hdf5 files.
            transform_feature (bool, optional): Scale depth and rgb images. Defaults to
            True.
            transform_label (bool, optional): Covert classes to one-hote encoding.
            Defaults to True.
            dtype (torch.dtype, optional): Dtype of all feature and lable data that is
            in float type. Defaults to torch.float32.
        """
        self.root_path = root_path
        self.hdf5_dir = hdf5_dir
        self.transform_feature = transform_feature
        self.transform_label = transform_label
        self.image_size = image_size
        self.dtype = dtype

        with open(os.path.join(self.root_path, coco_annotations_file)) as f:
            annotations = json.load(f)["annotations"]
            self.annotations = defaultdict(list)
            for annotation in annotations:
                if self.transform_label:
                    category = annotation["category_id"] - 1

                else:
                    if utils.CLASS_MAP["small"] == annotation["category_id"]:
                        category = "small"
                    elif utils.CLASS_MAP["medium"] == annotation["category_id"]:
                        category = "medium"
                    else:
                        category = "unkonwn"
                        print(f"Unknown category {annotation["category_id"]=}")

                self.annotations[annotation["image_id"]].append(
                    (
                        category,
                        annotation["bbox"],
                        annotation["segmentation"],
                    )
                )

    def __len__(self) -> None:
        return len(os.listdir(os.path.join(self.root_path, self.hdf5_dir)))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Get dataset item at idx.

        Args:
            idx (int): Index.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing feature tensor and label dict.
        """
        with h5py.File(os.path.join(self.root_path, self.hdf5_dir, f"{idx}.hdf5")) as f:
            rgb_img = torch.tensor(np.array(f["colors"]))
            rgb_img = rgb_img.permute(2, 0, 1)

            depth_img = torch.tensor(np.array(f["depth"]))

            obj_infos = json.loads(np.array(f["instance_attribute_maps"]).tobytes())

        H, W = depth_img.shape

        if H < W:
            pad_top = (W - H) // 2
            pad_bottom = (W - H) - pad_top
            pad_left = pad_right = 0
        elif W < H:
            pad_left = (H - W) // 2
            pad_right = (H - W) - pad_left
            pad_top = pad_bottom = 0
        else:
            pad_top = pad_bottom = pad_left = pad_right = 0

        padding_tuple = (pad_left, pad_top, pad_right, pad_bottom)

        depth_img = to_dtype(
            depth_img.unsqueeze_(0),
            dtype=self.dtype,
            scale=self.transform_feature,
        )
        depth_img = pad(depth_img, padding_tuple)
        depth_img = resize(depth_img, (self.image_size, self.image_size))

        rgb_img = to_dtype(
            rgb_to_grayscale(
                rgb_img,
            ),
            dtype=self.dtype,
            scale=self.transform_feature,
        )
        rgb_img = pad(rgb_img, padding_tuple)
        rgb_img = resize(rgb_img, (self.image_size, self.image_size))

        X = torch.cat([rgb_img, depth_img])

        positions = []
        rot6ds = []
        for info in obj_infos:
            if info["idx"] != 0:
                positions.append(info["position"])

                if self.transform_label:
                    rot6ds.append(utils.from_euler_to_rot6d(info["euler"]))
                else:
                    rot6ds.append(info["euler"])
        positions = torch.tensor(positions, dtype=self.dtype)
        rot6ds = torch.tensor(rot6ds, dtype=self.dtype)

        if self.transform_label:
            categories = nn.functional.one_hot(
                torch.tensor(
                    [instance[0] for instance in self.annotations[idx]]
                ).long(),
                2,
            )
        else:
            categories = [[instance[0]] for instance in self.annotations[idx]]

        bboxes = torch.tensor(
            [instance[1] for instance in self.annotations[idx]], dtype=self.dtype
        )
        bboxes[:, :2] = bboxes[:, :2] + torch.tensor(padding_tuple[:2])
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * (self.image_size / max(H, W))
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * (self.image_size / max(H, W))
        bboxes[:, [0, 2]] /= rgb_img.shape[2]
        bboxes[:, [1, 3]] /= rgb_img.shape[1]

        masks = [instance[2] for instance in self.annotations[idx]]
        masks = torch.tensor(
            np.array(
                [
                    maskUtils.decode(
                        maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
                    )
                    for rle in masks
                ]
            )
        )
        masks = pad(masks, padding_tuple)
        masks = resize(masks, (self.image_size, self.image_size))

        y = {
            "positions": positions,
            "rot6ds": rot6ds,
            "categories": categories,
            "bboxes": bboxes,
            "masks": masks,
        }

        return X, y

    @staticmethod
    def visualize_dataset(
        training_data: Dataset,
        start: int,
        stop: int = None,
        options: set = {"color", "bboxes", "seg_masks", "depth"},
    ) -> None:
        """Visualizes the dataset between start (incl) and stop (excl).

        Args:
            training_data (Dataset): Pytorch dataset object containing you the data.
            start (int): Start index.
            end (int, optional): Stop Index. If no index is provided, then the data at
            start is visualized.
            options (set, optional): Elements to visualize. Defaults to {"color", "bboxes",
            "seg_masks", "depth"}.
        """
        if stop == None:
            stop = start + 1

        if start == -1 and stop == -1:
            gen = range(0, len(training_data))
        elif start == -1:
            gen = range(0, stop)
        elif stop == -1:
            gen = range(start, len(training_data))
        else:
            gen = range(start, stop)

        for i in gen:
            X, y = training_data[i]
            utils.visualize_box_and_pose_data(X, y, options)


if __name__ == "__main__":
    training_data = BoxAndPoseDataset(
        "data/synthetic_data",
        "coco_annotations.json",
        "hdf5",
        transform_feature=True,
        transform_label=True,
        dtype=torch.float32,
    )

    print(len(training_data))

    sample = training_data[0]
    print(sample[0])
    print(sample[0].type())
    print(sample[0].shape)

    print(sample[1])

    for s in sample[1].values():
       print(s.shape)

    BoxAndPoseDataset.visualize_dataset(
        training_data, 1000, options={"color", "bboxes", "depth"}
    )
