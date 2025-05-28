import cv2
import tensorflow as tf
import json
import gc
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarkerOptions
import numpy as np

#loading files
video_path = '../Dataset/WSASL/test'
mediapipe_data_path = '../Dataset/hand_tracking.task'

with open('../Dataset/WSASL/Index.json', 'r') as f:
    data = json.load(f)

all_landmark = list()

global_framecount = 0
video_number = 1
#loading classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode

#config the landmarker
options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = mediapipe_data_path), running_mode = VisionRunningMode.VIDEO)
with tf.io.TFRecordWriter('../Dataset/tfrecords/test.tfrecord') as writer: # starts the loop to write the shit to disk
    with HandLandmarker.create_from_options(options) as landmarker: #initializes mp handlandmarker
        #frame splicing
        for x in data:
            instance = x['instances']
            gloss = x['gloss']
            for y in instance:
                    if y['split'] == 'test':
                        video_id = y['video_id']
                        all_landmark.clear()
                        vid = cv2.VideoCapture(f'{video_path}/{video_id}.mp4')
                        fps = vid.get(cv2.CAP_PROP_FPS)
                        while True:
                            ret, frame = vid.read() #honestly have no clue why ret still exists but it don't work without it so :Shrug:
                            if not ret:
                                break
                            else:
                                timestamp = int(1000*global_framecount/fps) #timestamp for mp(god i hated this.. ts took forever to get working)
                                frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting to Rgb just incase coz mp is a picky eater :/
                                mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame) #makes it something mp can process
                                del frame #we mark frame to be deleted from memory coz we aint needing it anymore
                                result = landmarker.detect_for_video(mp_image, timestamp)#we mark where the fingers are
                                global_framecount+=1#increase frame count for mp
                                if result.hand_landmarks: #checking if result has any data (otherwise it'll throw a fit)
                                    all_landmark.append([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])#stores the result for this frame
                                    del result#bye bye result hello memory
                        vid.release()# the video is a free man/woman again
                        lndmrk_arr = np.array(all_landmark) # appends all the frames into one ginormous array.. hmm wonder if it is taller than a trex
                        feature = { #does some hocus pocus that i do not fully yet understand but just go with coz it works (hopefully?)
                            'gloss': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gloss.encode()])), # the hocus pocus to store the video id
                            'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lndmrk_arr.tobytes()])), # the hocus pocus to store the array
                            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=(lndmrk_arr.shape))) # the hocus pocus to store the shape, idk why but guess its imp
                        } # hocus pocus over :(
                        example = tf.train.Example(features=tf.train.Features(feature=feature)) # just kidding.. its not.. :)
                        writer.write(example.SerializeToString()) # now it is frfr
                        del lndmrk_arr # bye bye list
                        gc.collect()
                        print(f'NO of videos done: {video_number}. {video_id} successfully written :thumbs_up:')
                        video_number +=1
                    else: continue

print('All Done niggah')
#if your reading this email me at abrahamkuruvila2008@proton.me with your favorite song right now.... pwease?