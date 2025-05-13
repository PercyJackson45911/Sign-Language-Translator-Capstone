import cv2
#import tensorflow as tf
import os
import json
import gc
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarkerOptions
import numpy as np
import pandas as pd
from tensorflow.python.ops.numpy_ops.np_utils import result_type

#loading files
video_path = '../Dataset/WSASL/temp'
mediapipe_data_path = '../Dataset/hand_tracking.task'

with open('../Dataset/WSASL/Index.json', 'r') as f:
    data = json.load(f)

array_dict = dict()
all_landmark = list()

global_framecount = 0
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
                    framecount = 0
                    while True:
                        ret, frame = vid.read()
                        if not ret:
                            break
                        #making it an image obj that mp can process
                        else:
                            timestamp = int(1000*global_framecount/fps)
                            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
                            del frame
                            result = landmarker.detect_for_video(mp_image, timestamp)
                            global_framecount+=1
                            if result.hand_landmarks:
                                temp = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
                                del result
                                all_landmark.append(temp)
                                del temp
                    vid.release()
                    array_dict[id] = np.array(all_landmark)
                    gc.collect()
                else: continue

data = list()
for id, landmarks in array_dict.items():
    for frame_landmarks in landmarks:
        data.append([id]+frame_landmarks.tolist())
df = pd.DataFrame(data, columns=["video_id", "landmark_x", "landmark_y", "landmark_z"])
print(df)
del df
del data
gc.collect()
