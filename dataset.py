import cv2
import json
import gc
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

#loading files
video_path = '/mnt/external/Capstone_Project/Dataset/WSASL/val'
mediapipe_data_path = 'hand_tracking.task'

with open('Index.json', 'r') as f:
    data = json.load(f)

all_landmark = list()
#loading classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#config the landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediapipe_data_path),
    running_mode=VisionRunningMode.VIDEO)

def frames():
    global_framecount = 0
    video_number = 1
    expected_frame = 0
    with HandLandmarker.create_from_options(options) as landmarker: #initializes mp handlandmarker.. or the stuff that finds the hand
        #frame splicing
        for x in data:
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
                            raise RuntimeError(f'Video {video_id} has no FPS info. MediaPipe will choke. Abort.')
                        while True:
                            ret, frame = vid.read() #honestly have no clue why ret still exists but it don't work without it so :Shrug:
                            if not ret:
                                if global_framecount==expected_frame:
                                    continue
                                else:
                                    print(f'video id {video_id} is corrupted.. please remove.. Danke')
                                break
                            else:
                                timestamp = int(1000*global_framecount/fps) #timestamp for mp(god i hated this.. ts took forever to get working)
                                frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting to Rgb just incase coz mp is a picky eater :/
                                mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame) #makes it something mp can process
                                del frame #we mark frame to be deleted from memory coz we aint needing it anymore
                                result = landmarker.detect_for_video(mp_image, timestamp)#we mark where the fingers are
                                global_framecount+=1#increase frame count for mp
                                frame_landmarks = []
                                for hand_landmarks in result.hand_landmarks:
                                    frame_landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                                while len(frame_landmarks)<42:
                                    frame_landmarks.append([0.0,0.0,0.0])
                                all_landmark.append(frame_landmarks)
                        vid.release()# the video is a free man/woman again
                        lndmrk_arr = np.array(all_landmark)# appends all the frames into one ginormous array.. hmm wonder if it is taller than a trex
                        gc.collect()
                        print(f'NO of videos done: {video_number}. {video_id} successfully written :thumbs_up:')
                        video_number +=1
                    else: continue
    print('frame processing done')
    return lndmrk_arr

print('All Done')
#if your reading this email me at abrahamkuruvila2008@proton.me with your favorite song right now.... pwease?
