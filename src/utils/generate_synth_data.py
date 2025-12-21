import blenderproc as bproc

import json
from pathlib import Path

import cv2
import numpy as np


CLASS_MAP = {
    "medium": 1,
    "small": 2,
}
OBJ_PATH = {
    "medium": "data/medium_box/medium_box_mesh.ply",
    "small": "data/small_box/small_box_mesh.ply",
}


def sample_pose(obj: bproc.types.MeshObject):
    n = 0
    while n < 100:
        u = np.random.uniform(0, image_width)
        v = np.random.uniform(0, image_height)
        z = np.random.choice(np.linspace(0.0, 0.01 * 110, num=111))

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        rot = np.random.choice([0, 1, 2, 3])
        diff = np.random.uniform(-0.1, 0.1)
        obj.set_location(
            (np.linalg.inv(world_to_cam_transform) @ np.array([x, y, z, 1]).T)[:3]
        )
        obj.set_rotation_euler([0, 0, rot * np.pi / 2 + diff])

        bbox = obj.get_bound_box()

        bbox = np.hstack([bbox, np.ones((bbox.shape[0], 1))]) @ world_to_cam_transform.T
        bbox = bbox[:, :3]

        bbox[:, 0] = fx * (bbox[:, 0] / bbox[:, 2]) + cx
        bbox[:, 1] = fy * (bbox[:, 1] / bbox[:, 2]) + cy

        valid = True
        for u, v, _ in bbox:
            if not ((0 < u < image_width) and (0 < v < image_height)):
                n += 1
                valid = False
                break

        if not valid:
            continue

        return
    print("No solution found")


bproc.init()

# Create a point light next to it
light = bproc.types.Light()
light.set_location([2, -2, 10])
light.set_energy(1000)

# Set the camera
with open("data/medium_box/intrinsics.json", "r") as f:
    cam_intrinsics = json.load(f)

K_matrix = np.eye(3)
K_matrix[0, 0] = cam_intrinsics["fx"]
K_matrix[1, 1] = cam_intrinsics["fy"]
K_matrix[0, 2] = cam_intrinsics["cx"]
K_matrix[1, 2] = cam_intrinsics["cy"]

image_width = cam_intrinsics["width"]
image_height = cam_intrinsics["height"]

bproc.camera.set_intrinsics_from_K_matrix(K_matrix, image_width, image_height)

fx, fy = K_matrix[0, 0], K_matrix[1, 1]
cx, cy = K_matrix[0, 2], K_matrix[1, 2]

with open("data/medium_box/cam2root.json", "r") as f:
    world_to_cam_transform = json.load(f)

world_to_cam_transform = (
    bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        world_to_cam_transform["cam2root"], ["X", "-Y", "-Z"]
    )
)
bproc.camera.add_camera_pose(world_to_cam_transform)
bproc.renderer.enable_depth_output(False)

# Create a simple object:
total_boxes = 6
dataset_size = 2
np.random.seed()
objs = []
for i in range(dataset_size):
    bproc.object.delete_multiple(objs)
    objs = []
    n_small_boxes = np.random.randint(0, total_boxes + 1)

    for j in range(1, n_small_boxes + 1):
        obj = bproc.loader.load_obj(OBJ_PATH["small"])
        obj[0].set_scale(np.full((3), 1 / 1000))
        obj[0].set_cp("category_id", CLASS_MAP["small"])
        obj[0].set_cp("instance_id", j)
        objs.extend(obj)

    for j in range(n_small_boxes + 1, total_boxes + 1):
        obj = bproc.loader.load_obj(OBJ_PATH["medium"])
        obj[0].set_scale(np.full((3), 1 / 1000))
        obj[0].set_cp("category_id", CLASS_MAP["medium"])
        obj[0].set_cp("instance_id", j)
        objs.extend(obj)

    bproc.object.sample_poses(
        objs, sample_pose_func=sample_pose, objects_to_check_collisions=objs
    )

    # Render the scene
    bproc.renderer.set_noise_threshold(0.1)
    bproc.renderer.enable_segmentation_output(
        map_by=["instance_id", "category_id"],
        default_values={"category_id": 0, "instance_id": 0},
    )
    data = bproc.renderer.render()

    data["depth"][0][data["depth"][0] > 10.0] = 0.0
    data["depth"][0] = (np.round(data["depth"][0] * 1000)).astype(np.uint16)

    # Write the rendering into an hdf5 file
    path = "data/synthetic_data/"
    bproc.writer.write_coco_annotations(
        path,
        instance_segmaps=data["instance_id_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )
    bproc.writer.write_hdf5(
        path + "/depth/", {"depth": data["depth"]}, append_to_existing_output=True
    )
