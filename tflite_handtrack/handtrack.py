"""
Custom script to perform hand tracking and optionally cropping and saving the bounding boxes content. I created it using
the following libraries/resources:

1. Video capture uses Opencv to stream from a webcam.
2. The detection utilizes a pre-trained Palm Detector model developed by Google AI Research,
   which was converted to a .tflite format for deployment on mobile devices. The model is available at:
   https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_detection_mobile_gpu.md
3. The handling of the Tensorflow Lite model is based on examples available at:
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/
"""

import os
import time
import argparse
import numpy as np
import cv2

from utils import preprocess_image, rescale_bbox
from detector import Detector

def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """

    # Define and parse input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--modeldir",
                        help="Folder the .tflite file is located.",
                        default="../tflite_model/")
    parser.add_argument("--graph",
                        help="Name of the .tflite file.",
                        default="palm_detection_without_custom_op.tflite")
    parser.add_argument("--labels",
                        help="Name of the labelmap file.",
                        default="palm_detection_labelmap.txt")
    parser.add_argument("--min_conf",
                        help="Minimum confidence threshold for displaying detected hand palm.",
                        type=float,
                        default=0.7)
    parser.add_argument("--input_filename",
                        help="Full filename of input file to process. Support formats: mp4, mp3, jpg, png",
                        required=True)

    parsed_args = parser.parse_args()

    return parsed_args


def main():

    args = parse_arguments()

    input_filename = args.input_filename

    if os.splittext(input_filename)[1] in ["mp4", "mp3"]:
        pass

    elif os.splittext(input_filename)[1] in ["jpg", "png"]:
        pass
    else:
        raise RuntimeError("Format of input file is not supported")


if __name__ == "__main__":
    raise NotImplementedError("Implementation pending")


