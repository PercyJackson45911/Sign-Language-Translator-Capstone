import cv2
import json
import gc
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from tqdm import tqdm
import torch
import os

# loading files
video_path = '/mnt/external/Capstone_Project/Dataset/WSASL/val'
mediapipe_data_path = 'hand_tracking.task'

with open('Index.json', 'r') as f:
    data = json.load(f)

all_landmark = list()
# loading classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# config the landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediapipe_data_path),
    running_mode=VisionRunningMode.VIDEO)

global_framecount = 0
video_number = 1
expected_frame = 0
failed = list()

# count number of .mp4 videos for progress bar
video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]

with HandLandmarker.create_from_options(options) as landmarker:  # initializes mp handlandmarker.. or the stuff that finds the hand
    # frame splicing
    for x in tqdm(data, desc='Processing videos', total=len(video_files)):
        instance = x['instances']
        gloss = x['gloss']
        for y in instance:
            if y['split'] == 'val':
                video_id = y['video_id']
                all_landmark.clear()
                vid = cv2.VideoCapture(f'{video_path}/{video_id}.mp4')
                fps = vid.get(cv2.CAP_PROP_FPS)
                expected_frame += vid.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps == 0:
                    failed.append(f'Video {video_id} has no FPS info.')
                    continue
                while True:
                    ret, frame = vid.read()  # honestly have no clue why ret still exists but it don't work without it so :Shrug:
                    if not ret:
                        if global_framecount == expected_frame:
                            break
                        else:
                            print(f'video id {video_id} is corrupted.. please remove.. Danke')
                            break
                    else:
                        timestamp = int(1000 * global_framecount / fps)  # timestamp for mp(god i hated this.. ts took forever to get working)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # converting to Rgb just incase coz mp is a picky eater :/
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  # makes it something mp can process
                        del frame  # we mark frame to be deleted from memory coz we aint needing it anymore
                        result = landmarker.detect_for_video(mp_image, timestamp)  # we mark where the fingers are
                        global_framecount += 1  # increase frame count for mp
                        frame_landmarks = []
                        for hand_landmarks in result.hand_landmarks:
                            frame_landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                        while len(frame_landmarks) < 42:
                            frame_landmarks.append([0.0, 0.0, 0.0])
                        all_landmark.append(frame_landmarks)
                vid.release()  # the video is a free man/woman again
                lndmrk_arr = np.array(all_landmark) # appends all the frames into one ginormous array.. hmm wonder if it is taller than a trex
                if np.all(lndmrk_arr == 0):  # <- was a bug here originally with `=` instead of `==`
                    failed.append(f'{video_id} is full of zeros.. check source')
                    continue
                else:
                    tensor = torch.tensor(lndmrk_arr, dtype = torch.float32)
                    data_dict = {'landmarks': tensor, 'gloss': gloss}
                    torch.save(data_dict, f'/mnt/external/Capstone_Project/Dataset/pt_files/{video_id}.pt')
                gc.collect()
                video_number += 1
            else:
                continue
    print('frame processing done')
    # return removed — you can't return outside a function lol
    # return lndmrk_arr, gloss

# save the failed video IDs to a text file
with open('failed.txt', 'w') as f:
    for x in failed:
        f.write(str(x)+'\n')

print('All Done')
#if your reading this email me at abrahamkuruvila2008@proton.me with your favorite song right now.... pwease?

