# smart_palletizer

![Detected boxes in image](docs/results/box_detection.gif)
![Spawned Boxes](docs/results/box_spawn.gif)

## Introduction

This repository applies computer vision algorithms to a ROS2 bag data-stream to process color and depth data, detect boxes placed on a pallet, and estimate poses.

## Instructions

### Installation

#### Docker (Recommended)
Docker is the recommended way to run this project. It ensures no issues related to the environment. You need to install [Docker](https://docs.docker.com/desktop/) on your PC. You can also follow these [instructions](https://docs.docker.com/engine/install/linux-postinstall/) post-install to avoid prefixing each docker command with `sudo`.

Clone this repository and put the unzipped [ROS2 compatible bag folder](https://drive.google.com/file/d/1kPUg90kEzcZHuLLqfFAULbLmw7cl4sGu/view?usp=sharing) in `data/`. Make sure to give access to your local display to run `rviz2`. On Linux, you can run:
```
xhost +local:root
```
In the root of the project:
```
docker compose run --rm smart_palletizer
```

#### Local PC
The project has been developed using Ubuntu 24.04 with ROS2 Kilted. It is recommended to use these versions as other combinations have not been tested. Make sure that you completely and accurately follow the [ROS2 Kileted install instructions](https://docs.ros.org/en/kilted/Installation/Ubuntu-Install-Debs.html). Make sure that you have `python` and `pip` installed.

Clone this repository and make sure you have sourced the ROS2 installation:
```
source /path/to/ros2/setup.bash
```
Usually, the path to ROS2 is `/opt/ros2/${ROS_DISTRO}`.

In the root folder of the cloned repository:
```
sudo apt-get update
sudo apt-get upgrade
rosdep init
rosdep update
rosdep install --from-paths src -yi
```
You might need to use `sudo` with `rosdep init`.

Build and source the built project:
```
colcon build 
source install/setup.bash
```

A [ROS2 compatible bag](https://drive.google.com/file/d/1kPUg90kEzcZHuLLqfFAULbLmw7cl4sGu/view?usp=sharing) file must be downloaded and extracted.

### Running
There are four convenient launch files that you can use to test and visualize the different features of the project.

- To see the results of post processing the point cloud:
    ```
    ros2 launch smart_palletizer_py post_processing.launch.py bag_path:=path/to/bag/folder/smart_palletizing_data_ros2/
    ```
- To see the results of box detection:
    ```
    ros2 launch smart_palletizer_py box_detection.launch.py bag_path:=path/to/bag/folder/smart_palletizing_data_ros2/
    ```
- To see the results of pose detection:
    ```
    ros2 launch smart_palletizer_py pose_detection.launch.py bag_path:=path/to/bag/folder/smart_palletizing_data_ros2/
    ```
- To see the spawned boxes according to detected poses:
    ```
    ros2 launch smart_palletizer_py box_spawn.launch.py bag_path:=path/to/bag/folder/smart_palletizing_data_ros2/
    ```

## Method and Results

This section explains the methodology and shows visualizations for each feature.

### Post Processing
- Image and depth data were downsampled by a factor of 2 using the median value of each subsampling array. This allowed faster processing and improved rviz2 performance during visualization.
- Holes (`val=0`) in the depth image were patched using the minimum value among the 3x1 column to the left of each pixel. This hole patching algorithm was run multiple times to patch thicker hole regions.
- Median blur was applied to the depth image to reduce noise without losing edges. A bidirectional EMA spatial filter was also tested, but due to its iterative nature (and the code being written in Python), the temporal performance was poor.
- The depth map was passed through an EMA time filter to further minimize noise.

The top image shows the unfiltered data, while the bottom one shows the filtered result.

![Unfiltered depth cloud](docs/results/post_processing_unfiltered.gif)
![Filtered depth cloud](docs/results/post_processing_filtered.gif)

### Box Detection
- The filtered color and depth images from the Post Processing node were taken. The color image was converted to HSV format as hues are distinguishable by the H value.
- Histograms of V values were equalized, resulting in sharper contrast in the complete HSV image.
- All 3 channels of the HSV image were passed through an EMA time filter to remove noise.
- A mask was created using the depth map to include only depths between the pallet and the top-most box.
- The HSV image was appended with the depth image channel. This combined image was masked using the mask.
- Contours were detected in two images: the combined image and the depth image. The combined image gave contours that were sufficiently differentiated in all 4 channels, while the contours from depth catered to objects whose contours were harder to detect due to interference from the HSV channels (for example, tapes and barcodes).
- Bounding boxes and minimum enclosing rectangles of contours were detected. The minimum enclosing rectangle was then used to get real-world lengths of the boxes. These lengths were compared to the reference lengths of the small and medium boxes. The class with minimal error was chosen as the class for the box.
- Information about the centroid and longest length of the boxes was published as a message for downstream tasks.

![Detected boxes in image](docs/results/box_detection.gif)

### Pose Detection
- The message containing information about the boxes was taken from the Box Detection node. This information contained centroid coordinates and the length of the longest side of the box.
- x-axis was assigned to the longest side, y-axis to the second longest and z-axis to the smallest in the reference dimensions. All boxes in the image can then be seen to have the z-axis pointing downwards (away from the picture). Therefore, the visible side shows the XY plane.
- The angle of the longest side was calculated with respect to the x-axis of the image. This angle was then converted to quaternion angles.
- The centroid and quaternion were used to broadcast the transform frame for each box.

![Detected poses](docs/results/pose_detection.gif)

> **Future:** Both sides of the detected contour (rather than just the longest side) should be compared to the visible box side. The algorithm will then be valid, even if the XY-plane condition does not hold.

### Box Spawn
- The detected pose from Pose Detection node was used to construct marker for each box.
- The scale of the marker was set according to the identified class of the box.

![Spawned Boxes](docs/results/box_spawn.gif)

> **Future:** The planar surfaces can be detected using algorithms like RANSAC which is more reliable.
