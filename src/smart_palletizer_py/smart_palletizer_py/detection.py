import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from smart_palletizer_py import utils


class Detection(Node):
    def __init__(self):
        super().__init__('detection')

        self.publisher_ = self.create_publisher(Image, '/img_with_detection', 10)
        self.publisher2_ = self.create_publisher(Image, '/img2', 10)

        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera_filtered/color/image_filtered',
            self.add_rgb_data,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera_filtered/aligned_depth_to_color/image_filtered',
            self.add_depth_data,
            10
        )

        self.timer = self.create_timer(0.1, self.detect_contours)

        self.bridge = CvBridge()
        self.combined_img = {'rgb': None, 'depth': None}
        self.prev_filtered_img = {'h': None, 's': None, 'v': None, 'depth': None}

    def add_rgb_data(self, msg):
        self.combined_img['rgb'] = msg

    def add_depth_data(self, msg):
        self.combined_img['depth'] = msg

    def detect_contours(self):
        if self.combined_img['rgb'] is None or self.combined_img['depth'] is None:
            return
        
        # Convert ROS2 images to OpenCV
        img_depth = self.bridge.imgmsg_to_cv2(self.combined_img['depth'], desired_encoding='16UC1')
        img_rgb = self.bridge.imgmsg_to_cv2(self.combined_img['rgb'], desired_encoding='bgr8')

        ### --- Depth image pre-processing --- ###
        img_depth_processed = img_depth

        ## Apply adaptive sharpening to increase contrast in the depth map
        #img_depth_processed = cv2.normalize(img_depth_processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        #img_depth_processed = clahe.apply(img_depth_processed)

        ### --- BGR image pre-processing --- ###
        # Convert BGR image to HSV and offset the color since our target boxes are close to 0 value. Also equalize histograms in V space to imrpove contrast
        img_rgb_processed = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        img_rgb_processed[:, :, 0] = 100 + img_rgb_processed[:, :, 0]
        img_rgb_processed[:, :, 2] = cv2.equalizeHist(img_rgb_processed[:, :, 2])

        # Apply time filtering to each channel
        self.prev_filtered_img['h'] = img_rgb_processed[:, :, 0] = utils.time_filter(img_rgb_processed[:, :, 0], self.prev_filtered_img['h'], 0.05, 180).copy()
        self.prev_filtered_img['s'] = img_rgb_processed[:, :, 1] = utils.time_filter(img_rgb_processed[:, :, 1], self.prev_filtered_img['s'], 0.1, 50).copy()
        self.prev_filtered_img['v'] = img_rgb_processed[:, :, 2] = utils.time_filter(img_rgb_processed[:, :, 2], self.prev_filtered_img['v'], 0.2, 50).copy()

        ### --- Combined image pre-processing --- ###
        # Create masks from depth to isolate just the region of interest
        mask1 = np.zeros_like(img_depth_processed)
        _, mask1 = cv2.threshold(img_depth_processed, 1775, 255, cv2.THRESH_BINARY_INV)
        mask1 = mask1.astype(np.uint8)

        mask2 = np.zeros_like(img_depth_processed)
        _, mask2 = cv2.threshold(img_depth_processed, 1500, 255, cv2.THRESH_BINARY)
        mask2 = mask2.astype(np.uint8)

        # Convert depth image to uint8 and append to the HSV image
        img_depth_processed = cv2.convertScaleAbs(img_depth_processed, alpha=255.0 / np.max(img_depth_processed))
        img_rgb_processed = np.dstack((img_rgb_processed, img_depth_processed))

        # Apply mask to combined image
        img_rgb_processed = cv2.bitwise_and(img_rgb_processed, img_rgb_processed, mask=mask1)
        img_rgb_processed = cv2.bitwise_and(img_rgb_processed, img_rgb_processed, mask=mask2)

        ### --- Edge generation, contour detection, post-processing and drawing --- ###
        # Isolate depth image
        img_depth_processed = img_rgb_processed[:, :, 3]

        # Run Canny edge detection on combined image to extract boxes that can be identified by both hsv and depth change
        contours, _ = self.getContours(img_rgb_processed, canny_range=(80, 100), dilation_kernel_size=(5, 5), dilation_iterations=1)
        # Run Canny edge detection on depth image to extract boxes that can be identified by only depth change
        contours_depth, _ = self.getContours(img_depth_processed, canny_range=(6, 6), dilation_kernel_size=(5, 5), dilation_iterations=2)

        # Draws the area representing the contour, bounding box and label
        img_rgb = self.drawDetectionResults(img_rgb, img_depth, contours_depth + contours)

        # Publish result
        header = Header()
        header.frame_id = "camera_color_optical_frame"
        header.stamp = self.get_clock().now().to_msg()
        
        img_msg = self.bridge.cv2_to_imgmsg(img_rgb, 'passthrough', header)
        self.publisher_.publish(img_msg)

        #img_msg = self.bridge.cv2_to_imgmsg(self.img_processed3, encoding='passthrough')
        #self.publisher2_.publish(img_msg)

        # Reset images
        self.combined_img['rgb'] = None
        self.combined_img['depth'] = None
        
        
        #self.img_processed3 = img

    def getContours(self, img, canny_range, dilation_kernel_size, dilation_iterations):
        img = cv2.Canny(img, canny_range[0], canny_range[1])

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=dilation_iterations)

        return cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    def drawDetectionResults(self, img_rgb: np.ndarray, img_depth: np.ndarray, contours: tuple):
        # Create a detection mask to highlight boxes with transparent color
        detection_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), dtype=img_rgb.dtype)
        selected_contours = []
        for cnt in contours:
            #depth = 0
            #for point in cnt:
            #    depth += img_depth[point[0][1], point[0][0]]
            #depth = round(depth/len(cnt))

            box = cv2.boxPoints(cv2.minAreaRect(cnt))
            box = np.int0(box)
            l1, l2 = utils.distance(box[0], box[1]), utils.distance(box[1], box[2])
            l1, l2 = max(l1, l2), min(l1, l2)
            ratio = l1 / l2
            area = l1 * l2

            #   ratios: x-y     y-z     z-x
            #   medium: 1.36    2.63    3.58
            #   small:  1.64    1.55    2.55
            #lim_low = (1.36 + 1.55)/2
            lim_low = 1.50
            lim_high = (2.63 + 2.55)/2
            diff = 0.1
            
            if lim_low <= ratio <= lim_high:
                text = "Small"
            elif 1.36 - diff < ratio < 3.58 + diff:
                text = "Medium"
            else:
                text = ""

            if 2800 < area < 15000 and text:
                #cv2.drawContours(img_rgb, np.array([box[0], box[1], [box[2]]]), -1, (0, 255, 0), 5)
                valid = True
                for cnt2 in selected_contours:
                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    if (
                        cnt2[0] <= cx <= cnt2[0] + cnt2[2] and
                        cnt2[1] <= cy <= cnt2[1] + cnt2[3]
                        ):
                        valid = False
                        break
                    
                if valid:
                    epsilon = 0.07 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    x, y, w, h = cv2.boundingRect(approx)
                    selected_contours.append((x, y, w, h))
                    
                    cv2.drawContours(detection_mask, [box], -1, (0, 0, 255), thickness=cv2.FILLED)
                    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 255, 255),)
                    cv2.putText(img_rgb, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
        return cv2.addWeighted(img_rgb, 0.8, detection_mask, 0.2, 0)

def main(args=None):
    rclpy.init(args=args)
    node = Detection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
