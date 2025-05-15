import cv2
import tensorflow as tf
import json
import gc
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarkerOptions
import numpy as np
from tensorflow.python.keras.models import model_from_json

mediapipe_data_path = '../Dataset/hand_tracking.task'

all_landmark = list()

cam = cv2.VideoCapture(0)
framecount = 0

#loading classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path = mediapipe_data_path), running_mode=VisionRunningMode.LIVE_STREAM)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        timestamp = int(1000*framecount/25)
        ret, frame = cam.read()
        start = time.time()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame)
        landmarker.detect_async(mp_image, timestamp)
        result = landmarker.detect_for_video(mp_image, timestamp)
        time.sleep(max(1./25 - (time.time() - start), 0))# leave this line at the end of the frame processing
        framecount += 1
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()