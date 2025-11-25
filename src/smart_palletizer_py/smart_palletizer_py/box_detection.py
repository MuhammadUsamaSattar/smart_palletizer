import math

import cv2
from cv_bridge import CvBridge
import image_geometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

from smart_palletizer_py import utils
from smart_palletizer_interfaces.msg import DetectedBox, DetectedBoxes


class BoxDetection(Node):
    def __init__(self):
        super().__init__("detection")

        self.detected_box_img_publisher_ = self.create_publisher(
            Image, "/box_detection_img", 10
        )
        self.detected_boxes_publisher_ = self.create_publisher(
            DetectedBoxes, "/detected_boxes", 10
        )

        self.rgb_subscription = self.create_subscription(
            Image, "/camera_filtered/color/image_filtered", self.add_rgb_data, 10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/camera_filtered/aligned_depth_to_color/camera_info",
            self.add_camera_info,
            10,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            "/camera_filtered/aligned_depth_to_color/image_filtered",
            self.add_depth_data,
            10,
        )

        self.timer = self.create_timer(1 / 30, self.detect_boxes)

        self.bridge = CvBridge()
        self.combined_img = {"rgb": None, "depth": None}
        self.prev_filtered_img = {"h": None, "s": None, "v": None, "depth": None}
        self.camera = image_geometry.PinholeCameraModel()

    def add_rgb_data(self, msg):
        self.combined_img["rgb"] = msg

    def add_depth_data(self, msg):
        self.combined_img["depth"] = msg

    def add_camera_info(self, msg):
        self.camera.from_camera_info(msg)

    def detect_boxes(self):
        if self.combined_img["rgb"] is None or self.combined_img["depth"] is None or self.camera.get_tf_frame is None:
            return

        # Convert ROS2 images to OpenCV
        img_depth = self.bridge.imgmsg_to_cv2(
            self.combined_img["depth"], desired_encoding="16UC1"
        )
        img_rgb = self.bridge.imgmsg_to_cv2(
            self.combined_img["rgb"], desired_encoding="bgr8"
        )

        ### --- Depth image pre-processing --- ###
        img_depth_processed = img_depth

        ## Apply adaptive sharpening to increase contrast in the depth map - MIGHT BE IMPLEMENTED LATER
        # img_depth_processed = cv2.normalize(img_depth_processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # img_depth_processed = clahe.apply(img_depth_processed)

        ### --- BGR image pre-processing --- ###
        # Convert BGR image to HSV and offset the color since our target boxes are close to 0 value. Also equalize histograms in V space to imrpove contrast
        img_rgb_processed = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        img_rgb_processed[:, :, 0] = 100 + img_rgb_processed[:, :, 0]
        img_rgb_processed[:, :, 2] = cv2.equalizeHist(img_rgb_processed[:, :, 2])

        # Apply time filtering to each channel
        self.prev_filtered_img["h"] = img_rgb_processed[:, :, 0] = utils.time_filter(
            img_rgb_processed[:, :, 0], self.prev_filtered_img["h"], 0.05, 180
        ).copy()
        self.prev_filtered_img["s"] = img_rgb_processed[:, :, 1] = utils.time_filter(
            img_rgb_processed[:, :, 1], self.prev_filtered_img["s"], 0.1, 50
        ).copy()
        self.prev_filtered_img["v"] = img_rgb_processed[:, :, 2] = utils.time_filter(
            img_rgb_processed[:, :, 2], self.prev_filtered_img["v"], 0.2, 50
        ).copy()

        ### --- Combined image pre-processing --- ###
        # Create masks from depth to isolate just the region of interest
        upper_depth_lim = 1775
        lower_depth_lim = 1500
        mask1 = np.zeros_like(img_depth_processed)
        _, mask1 = cv2.threshold(
            img_depth_processed, upper_depth_lim, 255, cv2.THRESH_BINARY_INV
        )
        mask1 = mask1.astype(np.uint8)

        mask2 = np.zeros_like(img_depth_processed)
        _, mask2 = cv2.threshold(
            img_depth_processed, lower_depth_lim, 255, cv2.THRESH_BINARY
        )
        mask2 = mask2.astype(np.uint8)

        # Convert depth image to uint8 and append to the HSV image
        np.clip(
            img_depth_processed, lower_depth_lim, upper_depth_lim, img_depth_processed
        )
        img_depth_processed = cv2.convertScaleAbs(
            img_depth_processed,
            alpha=255.0 / (upper_depth_lim - lower_depth_lim),
            beta=-lower_depth_lim,
        )
        img_rgb_processed = np.dstack((img_rgb_processed, img_depth_processed))

        # Apply mask to combined image
        img_rgb_processed = cv2.bitwise_and(
            img_rgb_processed, img_rgb_processed, mask=mask1
        )
        img_rgb_processed = cv2.bitwise_and(
            img_rgb_processed, img_rgb_processed, mask=mask2
        )

        ### --- Edge generation, contour detection, post-processing and drawing --- ###
        # Isolate depth image
        img_depth_processed = img_rgb_processed[:, :, 3]

        # Run Canny edge detection on combined image to extract boxes that can be identified by both hsv and depth change
        contours, _ = self.findContours(
            img_rgb_processed,
            canny_range=(110, 110),
            dilation_kernel_size=(7, 7),
            dilation_iterations=1,
        )
        # Run Canny edge detection on depth image to extract boxes that can be identified by only depth change
        contours_depth, _ = self.findContours(
            img_depth_processed,
            canny_range=(40, 40),
            dilation_kernel_size=(5, 5),
            dilation_iterations=2,
        )

        # Draws the area representing the contour, bounding box and label
        detected_boxes = self.findBoxes(img_depth, contours_depth + contours)
        img_rgb = self.drawBoxes(img_rgb, detected_boxes, alpha=0.2)

        # Publish result
        header = Header()
        header.frame_id = "camera_color_optical_frame"
        header.stamp = self.get_clock().now().to_msg()

        img_msg = self.bridge.cv2_to_imgmsg(img_rgb, "passthrough", header)
        self.detected_box_img_publisher_.publish(img_msg)

        img_msg = self.bridge.cv2_to_imgmsg(self.img_processed2, encoding="passthrough")

        detected_boxes_data = []
        for _, _, box_info in detected_boxes:
            x, y, longest_side_coords, depth, classification = box_info
            detected_box = DetectedBox()
            detected_box.x = int(x)
            detected_box.y = int(y)
            detected_box.longest_side_coords[0].x = int(longest_side_coords[0][0])
            detected_box.longest_side_coords[0].y = int(longest_side_coords[0][1])
            detected_box.longest_side_coords[1].x = int(longest_side_coords[1][0])
            detected_box.longest_side_coords[1].y = int(longest_side_coords[1][1])
            detected_box.depth = int(depth)
            detected_box.classification = classification
            detected_boxes_data.append(detected_box)

        detected_boxes_msg = DetectedBoxes()
        detected_boxes_msg.header = header
        detected_boxes_msg.data = detected_boxes_data
        self.detected_boxes_publisher_.publish(detected_boxes_msg)

        # Reset images
        self.combined_img["rgb"] = None
        self.combined_img["depth"] = None

    def findContours(self, img, canny_range, dilation_kernel_size, dilation_iterations):
        img = cv2.Canny(img, canny_range[0], canny_range[1])

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        img = cv2.morphologyEx(
            img, cv2.MORPH_DILATE, kernel, iterations=dilation_iterations
        )
        self.img_processed2 = img

        return cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    def findBoxes(self, img_depth: np.ndarray, contours: tuple):
        # Create a detection mask that highlights detected boxes with transparent color
        detected_boxes = []
        for cnt in contours:
            depth = np.mean([img_depth[point[0][1], point[0][0]] for point in cnt])

            box_contour = cv2.boxPoints(cv2.minAreaRect(cnt))
            box_contour = np.int0(box_contour)

            p1 = utils.get_XYZ_from_Pixels(self.camera, box_contour[0][0], box_contour[0][1], depth)
            p2 = utils.get_XYZ_from_Pixels(self.camera, box_contour[1][0], box_contour[1][1], depth)
            p3 = utils.get_XYZ_from_Pixels(self.camera, box_contour[2][0], box_contour[2][1], depth)
            l1, l2 = math.dist(p1, p2), math.dist(
                p2, p3
            )

            if l1 >= l2:
                longest_side_coords = [box_contour[0], box_contour[1]]
            else:
                l1, l2 = l2, l1
                longest_side_coords = [box_contour[1], box_contour[2]]

            box_dim = [[0.255, 0.155, 0.100], [0.340, 0.250, 0.095]]
            errors = [((l1-box_dim[0][0])**2 + (l2-box_dim[0][1])**2), ((l1-box_dim[1][0])**2 + (l2-box_dim[1][1])**2)]

            if errors[0] <= errors[1]:
                classification = "Small"
            else:
                classification = "Medium"
                errors[0], errors[1] = errors[1], errors[0]

            if errors[0] < 0.008:
                valid = True
                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                for prev_cnts in detected_boxes:
                    if (
                        (prev_cnts[0][0] <= cx <= prev_cnts[0][0] + prev_cnts[0][2]
                        and prev_cnts[0][1] <= cy <= prev_cnts[0][1] + prev_cnts[0][3])
                    ):
                        valid = False
                        break

                if valid:
                    epsilon = 0.07 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    bounding_box = cv2.boundingRect(approx)

                    detected_boxes.append(
                        (
                            bounding_box,
                            box_contour,
                            (cx, cy, longest_side_coords, depth, classification),
                        )
                    )

        return detected_boxes

    def drawBoxes(self, img, detected_boxes, alpha=0.2):
        detection_mask = np.zeros(img.shape, dtype=img.dtype)

        for bounding_box, box_contour, box_info in detected_boxes:
            x, y, w, h = bounding_box
            cv2.drawContours(
                detection_mask, [box_contour], -1, (0, 0, 255), thickness=cv2.FILLED
            )
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
            )
            cv2.rectangle(
                img,
                (x, y - 4),
                (x + (60 if box_info[4] == "Medium" else 44), y - 24),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                img,
                box_info[4],
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
            )

        return cv2.addWeighted(img, 1 - alpha, detection_mask, alpha, 0)


def main(args=None):
    rclpy.init(args=args)
    node = BoxDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
