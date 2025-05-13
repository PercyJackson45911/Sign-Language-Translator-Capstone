import cv2
#import tensorflow as tf
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarkerOptions
import numpy as np
from tensorflow.python.ops.numpy_ops.np_utils import result_type

#loading files
video_path = '../Dataset/WSASL/temp'
mediapipe_data_path = '../Dataset/hand_tracking.task'

with open('../Dataset/WSASL/Index.json', 'r') as f:
    data = json.load(f)

framecount = 0
#loading classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode

#config the landmarker
options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = mediapipe_data_path), running_mode = VisionRunningMode.VIDEO)
with HandLandmarker.create_from_options(options) as landmarker:
    #frame splicing
    for x in data:
        instance = x['instances']
        for y in instance:
                if y['split'] == 'train':
                    id = y['video_id']
                    vid = cv2.VideoCapture(f'{video_path}/{id}.mp4')
                    fps = vid.get(cv2.CAP_PROP_FPS)
                    while True:
                        ret, frame = vid.read()
                        if not ret:
                            break
                        #making it an image obj that mp can process
                        else:
                            timestamp = int(1000*framecount/fps)
                            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
                            result = landmarker.detect_for_video(mp_image, timestamp)
                            framecount+=1
                    vid.release()
                else: continue

