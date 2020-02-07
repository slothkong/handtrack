import csv
import numpy as np
import tensorflow as tf

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

class Detector:

    def __init__(self, model_path, labels_path, anchors_path):

        self.labels = self.load_labelmap(labels_path)
        self.anchors = self.load_anchors(anchors_path)

        self.interpreter = self.load_model(model_path)

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]

    # ------------------------------------------------------------------------------------------------------------------
    # Main method for inference
    # ------------------------------------------------------------------------------------------------------------------

    def detect(self, normalized_image, min_confidence=0.7):
        """ Perform inference

        :param normalized_image:
        :param min_confidence:
        :return: Dictionary with bounding box definition of most confidant detection
        """

        self.interpreter.set_tensor(self.input_details[0]['index'], normalized_image[None])
        self.interpreterinterpreter.invoke()

        bboxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0, :, 0]

        canditate_indices = (1 / (1 + np.exp(-scores))) > min_confidence
        candidate_detections = bboxes[canditate_indices]
        candidate_anchors = self.anchors[canditate_indices]

        if candidate_detections.shape[0] == 0:
            print("Failed to produce detection with the desired confidence")
            return None

        print("Successfully produced {} detections ".format(candidate_detections.shape[0]))

        best_candidate_index = np.argmax(candidate_detections[:, 3])
        dx, dy, w, h = candidate_detections[best_candidate_index, :4]
        centroid = candidate_anchors[best_candidate_index, :2] * 256

        return {"dx": dx, "dy": dy, "w": w, "h": h, "centroid": centroid}

    # ------------------------------------------------------------------------------------------------------------------
    # Auxiliary methods for data loading
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def load_model(model_path):
        """
        Load the Tensorflow Lite model.
        :return: Interpreter object
        """
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    @staticmethod
    def load_labelmap(labels_path):
        """
        Load the label map
        :return: List of labels
        """
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]

        return labels

    @staticmethod
    def load_anchors(anchors_path):
        """
        Load anchor definitions to be used by the SSD model
        :param anchors_path:
        :return:
        """

        with open(anchors_path, "r") as csv_f:
            anchors_list = [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            anchors = np.vstack(anchors_list)

        return anchors
