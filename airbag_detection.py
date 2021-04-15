import datetime
import os
import threading
import time
import sys

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from fastai.vision.all import *

import constants
from constants import *


class Airbag_Deployment_Detection:
    def __init__(self):
        self.video_stream = None
        self.input_video_file_name = video_file_name
        self.airbag_prediction_model = None
        self.frame = None
        self.class_idx = None
        self.prediction = None
        self.probability = None
        self.run_program = RUN_PROGRAM
        self.text = None
        self.colorIndex = None
        self.full_video_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.input_video_file_name)

        self.initialize_camera()
        self.load_pytorch_model()

    def initialize_camera(self):
        """
        This function will initialize the camera or video stream by figuring out whether to stream the camera capture or from a video file.
        :key
        """
        if self.input_video_file_name is None:
            print("[INFO] starting a camera video stream...")
            self.video_stream = VideoStream(src=constants.VID_CAM_INDEX).start()
            print("[INFO] Waiting 2 Seconds...")
            time.sleep(2)
        else:
            print("[INFO] Reading the video file...")
            self.video_stream = cv2.VideoCapture(self.input_video_file_name)

    def load_pytorch_model(self):
        """
        This function will load the CNN Classifier used for predicting if the frame has an opened airbag or not.
        :return:
        """
        print("[INFO] Loading the CNN Classifier for Airbag Prediction...")
        self.airbag_prediction_model = load_learner(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            model_name), cpu=True)
        self.airbag_prediction_model.eval()

    def grab_next_frame(self):
        """
        This function extracts the next frame from the video stream.
        :return:
        """
        if self.input_video_file_name is None:
            self.frame = self.video_stream.read()
        else:
            _, self.frame = self.video_stream.read()
        if self.frame is None:
            return

    def predict_on_frame(self, img):
        """
        This function will predict on any given image using the CNN classifier
        :param img:
        :return:
        """
        img = cv2.resize(img, (model_input_size, model_input_size))
        self.prediction, self.class_idx, self.probability = self.airbag_prediction_model.predict(img)

    def create_frame_icons(self):
        """
        This function will create the icons that will be displayed on the frame.
        :return:
        """
        self.text = "{}".format(self.prediction)
        self.colorIndex = LABELS.index(self.prediction)
        self.color = COLORS[self.colorIndex]

    def loop_over_frames(self):
        """
        This function is the main function that will loop through the frames and predict if the airbag is deployed or not.
        :return:
        """
        while self.run_program:
            self.grab_next_frame()
            self.predict_on_frame(img=self.frame)
            self.create_frame_icons()
            if self.class_idx == AIRBAG_PREDICTION_INDEX:
                print("[Prediction] An airbag has deployed!")
                cv2.putText(self.frame, self.text, (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
            if self.class_idx == NON_AIRBAG_PREDICTION_INDEX:
                cv2.putText(self.frame, self.text, (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
            cv2.imshow("video_frame", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


if __name__ == "__main__":
    airbag_detection_inst = Airbag_Deployment_Detection()
    airbag_detection_inst.loop_over_frames()
