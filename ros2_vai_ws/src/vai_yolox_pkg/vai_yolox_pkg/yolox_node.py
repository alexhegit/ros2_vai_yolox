#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import pathlib
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CURRENT_DIR = pathlib.Path(__file__).parent
sys.path.append(str(CURRENT_DIR))

from coco import COCO_CLASSES
from demo_utils import mkdir, multiclass_nms, demo_postprocess, vis


class VAI_YOLOX(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("AMD VAI YOLOX inference node init")
        self.declare_parameters(
            namespace = '',
            parameters = [
                ("model", "/tmp/yolox-s-int8.onnx"),
                ("vaip_config", "/tmp/vaip_config.json"),
                ("score_thr", 0.3),
                ("output_dir", "/tmp/demo_output"),
                ("image_path", "yolox-demo.jpg"),
                ("ipu", "True"),
                ("s_qos", 10, ),
                ("p_qos", 10)
            ]
        )
        self.score_thr = self.get_parameter("score_thr")
        self.output_dir = self.get_parameter("output_dir")
        self.image_path = self.get_parameter("image_path")
        self.model = self.get_parameter("model")
        self.ipu = self.get_parameter("ipu")
        self.vaip_config = self.get_parameter("vaip_config")
        self.s_qos = self.get_parameter("s_qos").get_parameter_value().integer_value
        self.p_qos = self.get_parameter("p_qos").get_parameter_value().integer_value
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(
            Image,
            '/image_raw',
            self.img_infer,
            self.s_qos)
        self.pub_img = self.create_publisher(Image, 'image_infer', self.p_qos)

    def img_infer(self, msg):
        # Convert ROS Image message to OpenCV image
        origin_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #origin_img = cv2.imread(self.image_path)

        # Preprocess the image here
        input_shape = (640, 640)  # replace with your actual input shape
        img, ratio = self.preprocess(origin_img, input_shape)

        # You can now use padded_img in your inference code...
        if self.ipu == "True":
            #self.get_logger().info("IPU EP")
            providers = ["VitisAIExecutionProvider"]
            provider_options = [{"config_file": self.vaip_config}]
        else:
            self.get_logger().info("CPU EP")
            providers = ['CPUExecutionProvider']
            provider_options = None
        session = ort.InferenceSession(self.model, providers=providers, provider_options=provider_options)
        # ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        ort_inputs = {session.get_inputs()[0].name: np.transpose(img[None, :, :, :], (0, 2 ,3, 1))}
        outputs = session.run(None, ort_inputs)
        outputs = [np.transpose(out, (0, 3, 1, 2)) for out in outputs]
        dets = self.postprocess(outputs, input_shape, ratio)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=self.score_thr, class_names=COCO_CLASSES)
            frame = cv2.resize(origin_img, (640, 480))
		    # opencv mat ->  ros msg
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub_img.publish(msg)

        mkdir(self.output_dir)
        output_path = os.path.join(self.output_dir, os.path.basename(self.image_path))

        cv2.imwrite(output_path, origin_img)

    def preprocess(self, img, input_shape, swap=(2, 0, 1)):
        """
        Preprocessing part of YOLOX for scaling and padding image as input to the network.

        Args:
        img (numpy.ndarray): H x W x C, image read with OpenCV
        input_shape (tuple(int)): input shape of the network for inference
        swap (tuple(int)): new order of axes to transpose the input image

        Returns:
        padded_img (numpy.ndarray): preprocessed image to be fed to the network
        ratio (float): ratio for scaling the image to the input shape
        """
        if len(img.shape) == 3:
            padded_img = np.ones((input_shape[0], input_shape[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_shape, dtype=np.uint8) * 114
        ratio = min(input_shape[0] / img.shape[0], input_shape[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, ratio

    def postprocess(self, outputs, input_shape, ratio):
        """
        Post-processing part of YOLOX for generating final results from outputs of the network.

        Args:
            outputs (tuple(numpy.ndarray)): outputs of the detection heads with onnxruntime session
            input_shape (tuple(int)): input shape of the network for inference
            ratio (float): ratio for scaling the image to the input shape

        Returns:
            dets (numpy.ndarray): n x 6, dets[:,:4] -> boxes, dets[:,4] -> scores, dets[:,5] -> class indices
        """
        outputs = [out.reshape(*out.shape[:2], -1).transpose(0,2,1) for out in outputs]
        outputs = np.concatenate(outputs, axis=1)
        outputs[..., 4:] = sigmoid(outputs[..., 4:])
        predictions = demo_postprocess(outputs, input_shape, p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(args=None):
    print(str(CURRENT_DIR))
    rclpy.init(args=args)
    vai_detector = VAI_YOLOX("vai_yolox")
    rclpy.spin(vai_detector)
    vai_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

