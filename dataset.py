import gc
import cv2
import tensorflow as tf
import json
import mediapipe as mp
import numpy as np

# Load files
mediapipe_data_path = '../Dataset/hand_tracking.task'
with open('../Dataset/WSASL/Index.json', 'r') as f:
    data = json.load(f)

# Build class-to-index mapping
all_glosses = sorted(set(x['gloss'] for x in data))
gloss_to_id = {gloss: idx for idx, gloss in enumerate(all_glosses)}

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediapipe_data_path),
    running_mode=VisionRunningMode.VIDEO
)

def video_generator():
    global_framecount = 0
    video_number = 1
    with HandLandmarker.create_from_options(options) as landmarker:
        for x in data:
            for y in x['instances']:
                video_path = f"../WSASL/Dataset/{y['split']}"
                video_id = y['video_id']
                vid = cv2.VideoCapture(f'{video_path}/{video_id}.mp4')
                fps = vid.get(cv2.CAP_PROP_FPS)
                frames = []

                while True:
                    ret, frame = vid.read()
                    if not ret:
                        break
                    timestamp = int(1000 * global_framecount / fps)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    result = landmarker.detect_for_video(mp_image, timestamp)
                    global_framecount += 1
                    if result.hand_landmarks:
                        frames.append([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
                    if len(frames) >= 160:
                        break

                vid.release()

                # Pad or trim to 160
                if len(frames) < 160:
                    pad_len = 160 - len(frames)
                    frames.extend([[[0.0, 0.0, 0.0]] * 21] * pad_len)
                else:
                    frames = frames[:160]

                landmark_arr = np.array(frames, dtype=np.float32)

                yield {
                    'video_id': tf.constant(video_id),
                    'landmarks': tf.constant(landmark_arr),
                    'shape': tf.constant(landmark_arr.shape),
                    'split': tf.constant(y['split']),
                    'gloss': tf.constant(gloss_to_id[x['gloss']])
                }

                print(f'NO of videos done: {video_number}. {video_id} successfully written.')
                video_number += 1
                gc.collect()

# TF Dataset
dataset = tf.data.Dataset.from_generator(
    video_generator,
    output_signature={
        'video_id': tf.TensorSpec(shape=(), dtype=tf.string),
        'landmarks': tf.TensorSpec(shape=(160, 21, 3), dtype=tf.float32),
        'shape': tf.TensorSpec(shape=(3,), dtype=tf.int32),
        'split': tf.TensorSpec(shape=(), dtype=tf.string),
        'gloss': tf.TensorSpec(shape=(), dtype=tf.int32)
    }
)

training_dataset = dataset.filter(lambda x: x['split'] == 'train').shuffle(100).repeat(10).batch(1)
val_dataset = dataset.filter(lambda x: x['split'] == 'test').repeat(10).batch(1)
test_dataset = dataset.filter(lambda x: x['split'] == 'test').repeat(10).batch(1)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(160, 21, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(gloss_to_id))  # num classes
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training
model.fit(training_dataset, validation_data=val_dataset, epochs=10)
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('Test Accuracy =', test_acc)
