import blenderproc as bproc

from collections import namedtuple
import json
import math

import numpy as np


CLASS_MAP = {
    "medium": 1,
    "small": 2,
}

bproc.init()

plane = bproc.object.create_primitive("PLANE")
plane.set_location([0, 0.65, -1.1])
plane.set_scale([2, 2, 2])

# Create a simple object:
BoxObjects = namedtuple("BoxObjects", ["medium", "small"])
objs = BoxObjects(
    bproc.loader.load_obj("data/medium_box/medium_box_mesh.ply")[0],
    bproc.loader.load_obj("data/small_box/small_box_mesh.ply")[0],
)

objs.medium.set_scale(np.full((3), 1 / 1000))
objs.medium.set_cp("category_id", CLASS_MAP["medium"])
objs.small.set_scale(np.full((3), 1 / 1000))
objs.small.set_cp("category_id", CLASS_MAP["small"])

objs.medium.set_location([0, 0.75, 0.150])
objs.medium.set_rotation_euler([0, 0, math.pi])
objs.small.set_location([0.2, 0.45, 0.150])
objs.small.set_rotation_euler([0, 0, math.pi])

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
bproc.camera.set_intrinsics_from_K_matrix(
    K_matrix, cam_intrinsics["width"], cam_intrinsics["height"]
)

with open("data/medium_box/cam2root.json", "r") as f:
    cam_extrinsics = json.load(f)

cam_extrinsics = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
    cam_extrinsics["cam2root"], ["X", "-Y", "-Z"]
)
bproc.camera.add_camera_pose(cam_extrinsics)

# Render the scene
bproc.renderer.set_noise_threshold(0.1)
bproc.renderer.enable_depth_output(False)
bproc.renderer.enable_segmentation_output(default_values={"category_id": 0})
data = bproc.renderer.render()

data["depth"][0][data["depth"][0] > 10.0] = 0.0

# Write the rendering into an hdf5 file
# bproc.writer.write_hdf5("data/synthetic_data/", data)
bproc.writer.write_coco_annotations(
    "data/synthetic_data/",
    instance_segmaps=data["category_id_segmaps"],
    instance_attribute_maps=data["instance_attribute_maps"],
    colors=data["colors"],
    color_file_format="JPEG",
)
bproc.writer.write_coco_annotations(
    "data/synthetic_data/",
    instance_segmaps=data["category_id_segmaps"],
    instance_attribute_maps=data["instance_attribute_maps"],
    colors=data["depth"],
    color_file_format="PNG",
)
