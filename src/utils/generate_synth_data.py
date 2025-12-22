import blenderproc as bproc

import json
import math
import os
import sys
from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# Mapping of object categories to IDs
CLASS_MAP = {
    "medium": 1,
    "small": 2,
}

# File paths for object meshes
OBJ_PATH = {
    "medium": "data/medium_box/medium_box_mesh.ply",
    "small": "data/small_box/small_box_mesh.ply",
}


def get_camera_info() -> Tuple[np.ndarray, int, int, float, float, float, float, Dict]:
    """
    Load camera intrinsics and extrinsics from JSON files.

    Returns:
        K_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        image_width (int): Width of the camera image.
        image_height (int): Height of the camera image.
        fx (float): Focal length in x.
        fy (float): Focal length in y.
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.
        cam_to_world_transform (Dict): Camera-to-world transformation dictionary.
    """
    # Load camera intrinsics
    with open("data/medium_box/intrinsics.json", "r") as f:
        cam_intrinsics = json.load(f)

    K_matrix = np.eye(3)
    K_matrix[0, 0] = cam_intrinsics["fx"]
    K_matrix[1, 1] = cam_intrinsics["fy"]
    K_matrix[0, 2] = cam_intrinsics["cx"]
    K_matrix[1, 2] = cam_intrinsics["cy"]

    image_width = cam_intrinsics["width"]
    image_height = cam_intrinsics["height"]

    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]

    # Load camera extrinsics
    with open("data/medium_box/cam2root.json", "r") as f:
        cam_to_world_transform = json.load(f)

    return K_matrix, image_width, image_height, fx, fy, cx, cy, cam_to_world_transform


def gen_boxes(total_boxes: int, n_small_boxes: int) -> List[bproc.types.MeshObject]:
    """
    Generate a list of mesh objects with category and instance IDs.

    Args:
        total_boxes (int): Total number of objects to generate.
        n_small_boxes (int): Number of small boxes to include.

    Returns:
        List[bproc.types.MeshObject]: List of loaded mesh objects.
    """
    objs = []

    # Generate small boxes
    for j in range(1, n_small_boxes + 1):
        obj = bproc.loader.load_obj(OBJ_PATH["small"])
        obj[0].set_scale(np.full((3), 1 / 1000))
        obj[0].set_cp("category_id", CLASS_MAP["small"])
        obj[0].set_cp("instance_id", j)
        objs.extend(obj)

    # Generate medium boxes
    for j in range(n_small_boxes + 1, total_boxes + 1):
        obj = bproc.loader.load_obj(OBJ_PATH["medium"])
        obj[0].set_scale(np.full((3), 1 / 1000))
        obj[0].set_cp("category_id", CLASS_MAP["medium"])
        obj[0].set_cp("instance_id", j)
        objs.extend(obj)

    return objs


def sample_pose_func(
    i: int, image_width: int, image_height: int, world_to_cam_transform: np.ndarray
):
    """
    Returns a function to sample poses for objects at a specific frame.

    Args:
        i (int): Frame index.
        image_width (int): Width of the camera image.
        image_height (int): Height of the camera image.
        world_to_cam_transform (np.ndarray): 4x4 camera extrinsics matrix.

    Returns:
        Callable[[bproc.types.MeshObject], None]: Pose sampling function.
    """

    def sample_pose(obj: bproc.types.MeshObject):
        """
        Sample a valid pose for an object ensuring it is within the camera view.

        Args:
            obj (bproc.types.MeshObject): Object to sample pose for.
        """
        n = 0
        while n < 100:
            # Random pixel coordinates in image plane
            u = np.random.uniform(0, image_width)
            v = np.random.uniform(0, image_height)
            # Random depth along Z-axis
            z = np.random.randint(0, 112) * 0.01

            # Convert pixel coordinates to camera coordinates
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # Random rotation around Z-axis
            rot = np.random.choice([0, 1, 2, 3])
            diff = np.random.uniform(-0.1, 0.1)

            euler_world = [0, 0, rot * np.pi / 2 + diff]

            # Calculate rotation relative to the camera frame
            euler_camera = R.from_euler("XYZ", euler_world).as_matrix()
            euler_camera = world_to_cam_transform[:3, :3] @ euler_camera
            euler_camera = R.from_matrix(euler_camera).as_euler("XYZ")

            # Set object location and rotation at frame i
            obj.set_location(
                (np.linalg.inv(world_to_cam_transform) @ np.array([x, y, z, 1]).T)[:3],
                frame=i,
            )
            obj.set_rotation_euler(euler_world, frame=i)

            # Project bounding box to camera frame
            bbox = obj.get_bound_box()
            bbox = (
                np.hstack([bbox, np.ones((bbox.shape[0], 1))])
                @ world_to_cam_transform.T
            )
            bbox = bbox[:, :3]

            bbox[:, 0] = fx * (bbox[:, 0] / bbox[:, 2]) + cx
            bbox[:, 1] = fy * (bbox[:, 1] / bbox[:, 2]) + cy

            # Check if bounding box is fully within image
            valid = True
            for u_pixel, v_pixel, _ in bbox:
                if not (0 < u_pixel < image_width and 0 < v_pixel < image_height):
                    n += 1
                    valid = False
                    break

            if valid:
                print(euler_camera)
                obj.set_cp("position", [x, y, z], frame=i)
                obj.set_cp("euler", euler_camera, frame=i)
                return

        print("No solution found")

    return sample_pose


def sample_poses_for_frames(
    n_frames: int,
    objs: List[bproc.types.MeshObject],
    image_width: int,
    image_height: int,
    world_to_cam_transform: np.ndarray,
):
    """
    Sample poses for all objects across multiple frames.

    Args:
        dataset_size (int): Total number of frames.
        total_boxes (int): Total number of objects in each frame.
        objs (List[bproc.types.MeshObject]): List of objects to sample poses for.
        image_width (int): Width of the camera image.
        image_height (int): Height of the camera image.
        world_to_cam_transform (np.ndarray): 4x4 camera extrinsics matrix.
    """
    for j in range(n_frames):
        bproc.renderer.set_world_background(
            [np.random.normal(0.35, 0.15)] * 3, strength=1
        )
        sample_pose = sample_pose_func(
            j, image_width, image_height, world_to_cam_transform
        )
        bproc.object.sample_poses(
            objs,
            sample_pose_func=sample_pose,
            objects_to_check_collisions=objs,
        )
        bproc.camera.add_camera_pose(world_to_cam_transform, frame=j)


def render_frames() -> dict:
    """
    Render the current scene.

    Returns:
        dict: Dictionary containing rendered outputs (colors, depth, segmentation maps, etc.).
    """
    bproc.renderer.set_noise_threshold(0.1)
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "instance_id", "category_id", "position", "euler"],
        default_values={
            "instance_id": 0,
            "category_id": 0,
            "position": [0.0, 0.0, 0.0],
            "euler": [0.0, 0.0, 0.0],
        },
    )
    return bproc.renderer.render()


def write_data(data: dict):
    """
    Write rendered data to COCO annotations and HDF5 depth maps.

    Args:
        data (dict): Output from BlenderProc renderer.
    """
    # Clip, create floor, simulate azure noise and convert depth to millimeters
    data["depth"][0][data["depth"][0] > 10.0] = 0.0
    data["depth"][0][data["depth"][0] == 0.0] = data["depth"][
        0
    ].max() + np.random.normal(0.1, 0.025)
    bproc.postprocessing.add_kinect_azure_noise(
        data["depth"][0], missing_depth_darkness_thres=0
    )
    data["depth"][0] = (np.round(data["depth"][0] * 1000)).astype(np.uint16)

    path = "data/synthetic_data/"

    # Data to store in the hdf5 file
    obj_pose_data = [
        [
            {
                "idx": instance_dict["idx"],
                "instance_id": instance_dict["instance_id"],
                "position": list(instance_dict["position"]),
                "euler": list(instance_dict["euler"]),
            }
            for instance_dict in frame
        ]
        for frame in data["instance_attribute_maps"]
    ]

    # Write COCO annotations
    bproc.writer.write_coco_annotations(
        path,
        instance_segmaps=data["instance_id_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )

    # Write depth maps to HDF5
    bproc.writer.write_hdf5(
        path + "/hdf5/",
        {
            "depth": data["depth"],
            "instance_attribute_maps": obj_pose_data,
        },
        append_to_existing_output=True,
    )


class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    # Initialize BlenderProc
    bproc.init()

    # Get camera info
    K_matrix, image_width, image_height, fx, fy, cx, cy, cam_to_world_transform = (
        get_camera_info()
    )

    # Set camera intrinsics
    bproc.camera.set_intrinsics_from_K_matrix(K_matrix, image_width, image_height)

    # Convert camera coordinates
    world_to_cam_transform = (
        bproc.math.change_source_coordinate_frame_of_transformation_matrix(
            cam_to_world_transform["cam2root"], ["X", "-Y", "-Z"]
        )
    )

    # Create a point light
    light = bproc.types.Light()
    light.set_location(
        [
            world_to_cam_transform[0][3],
            world_to_cam_transform[1][3],
            world_to_cam_transform[3][3] + 1.0,
        ]
    )
    light.set_energy(1000)

    # Enable depth output
    bproc.renderer.set_render_devices(desired_gpu_device_type="CUDA")
    bproc.renderer.enable_depth_output(False)

    # Dataset generation parameters
    total_boxes = 10
    approx_dataset_size = 10_000
    minibatch_size = 100  # Number of frames/poses per minibatch

    objs = []
    # Compute total minibatches across all combinations
    combination_dataset_sizes = []

    for i in range(total_boxes + 1):
        combination_size = int(math.ceil(approx_dataset_size / (total_boxes + 1)))
        combination_dataset_sizes.append(combination_size)
    total_frames_overall = sum(combination_dataset_sizes)

    np.random.seed()
    # Outer progress bar for the complete dataset
    with tqdm(
        total=total_frames_overall, desc="Overall dataset progress", unit="frames"
    ) as pbar_overall:
        # Loop over all object combinations (i small boxes)
        for i, combination_dataset_size in enumerate(combination_dataset_sizes):
            # Delete previous objects
            bproc.object.delete_multiple(objs)

            # Generate objects for this combination
            objs = gen_boxes(total_boxes, i)

            # Number of minibatches for this combination
            num_minibatches = math.ceil(combination_dataset_size / minibatch_size)

            # Inner progress bar for this combination
            for mb in tqdm(
                range(num_minibatches),
                desc=f"Combination {i} ({i} small boxes - {total_boxes-i} medium boxes)",
                unit="minibatch",
                leave=False,  # Don't leave the inner progress bar after completion
            ):
                start_idx = mb * minibatch_size
                end_idx = min((mb + 1) * minibatch_size, combination_dataset_size)
                current_batch_size = end_idx - start_idx

                # Sample poses only for the current minibatch
                with SuppressOutput():
                    sample_poses_for_frames(
                        n_frames=current_batch_size,
                        objs=objs,
                        image_width=image_width,
                        image_height=image_height,
                        world_to_cam_transform=world_to_cam_transform,
                    )

                # Render frames for this minibatch
                data = render_frames()

                # Write rendered data
                write_data(data)

                # Update overall progress bar
                pbar_overall.update(current_batch_size)
