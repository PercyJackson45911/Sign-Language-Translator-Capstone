import gc
import cv2
import tensorflow as tf
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarkerOptions
import numpy as np

from sort import video_id

#loading files
mediapipe_data_path = '../Dataset/hand_tracking.task'

with open('../Dataset/WSASL/Index.json', 'r') as f:
    data = json.load(f)

all_landmark = list()
array_dict = {}
np.set_printoptions(precision=4)


#loading classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode

#config the landmarker
options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = mediapipe_data_path), running_mode = VisionRunningMode.VIDEO)
def video_generator():
    video_number = 1
    global_framecount = 0
    with HandLandmarker.create_from_options(options) as landmarker: #initializes mp handlandmarker
        #frame splicing
        for x in data:
            instance = x['instances']
            for y in instance:
                    if y['split'] == 'train':
                        video_path ='../WSASL/Dataset/train'
                    elif y['split'] == 'test':
                        video_path = '../WSASL/Dataset/test'
                    elif y['split'] == 'val':
                        video_path = '../WSASL/Dataset/val'
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
                    landmark_arr = np.array(all_landmark) # appends all the frames into one ginormous array.. hmm wonder if it is taller than a trex
                    sample = {
                        'video_id': tf.constant(video_id),
                        'landmarks': tf.constant(landmark_arr),  # converted to tensor
                        'shape': tf.constant(landmark_arr.shape),
                        'split': tf.constant(y['split']),
                        'gloss': tf.constant(y['gloss'])
                    }
                    yield sample
                    del landmark_arr
                    print(f'NO of videos done: {video_number}. {video_id} successfully written :thumbs_up:')
                    video_number +=1
                    gc.collect()

dataset = tf.data.Dataset.from_generator(
    video_generator,
    output_signature = {
        'video_id': tf.TensorSpec(shape = (), dtype = tf.string),
        'landmarks': tf.TensorSpec(shape = (None, 21, 3), dtype = tf.float32),
        'shape': tf.TensorSpec(shape = (3,), dtype = tf.int32),
        'split': tf.TensorSpec(shape = (), dtpe = tf.string),
        'gloss': tf.TensorSpec(shape = (), dtpe = tf.string)
    }
)

training_dataset = dataset.filter(lambda x: x['split'] == 'train')
testing_dataset = dataset.filter(lambda x: x['split'] == 'test')
val_dataset = dataset.filter(lambda x: x['split'] == 'val')

gloss_to_id = {}  # fill with your mapping
def encode(x):
    x['label'] = tf.constant(gloss_to_id[x['gloss'].numpy().decode()], dtype=tf.int32)
    return x
training_dataset = training_dataset.map(lambda x: tf.py_function(encode, [x], Tout={'video_id': tf.string, 'landmarks': tf.float32, 'shape': tf.int32, 'split': tf.string, 'gloss': tf.string, 'label': tf.int32}))

# Extract features and labels
train_ds = training_dataset.map(lambda x: (x['landmarks'], x['label']))

# Dummy model example (adjust input shape to match your data)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(None, 21, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(gloss_to_id))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds.batch(32), epochs=5)


print('All Done niggah')
#if your reading this email me at abrahamkuruvila2008@proton.me with your favorite song right now.... pwease?