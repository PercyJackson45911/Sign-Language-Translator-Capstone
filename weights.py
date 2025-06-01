import tensorflow as tf
import numpy as np
import keras
from keras import layers

record_filepath = '../Dataset/tfrecords/train.tfrecord'

record = tf.data.TFRecordDataset(record_filepath)
result = dict()
gloss_to_id = dict()
id_to_gloss = dict()
landmark_dict = dict()


def generator():
    id = 0
    for i in record:
        example = tf.train.Example()
        example.ParseFromString(i.numpy())
        result = {}
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            result[key] = getattr(feature, kind).value

        gloss = result['gloss'][0].decode('utf-8')
        shape = result['shape']
        landmarks_raw = result['landmarks'][0]
        landmarks = np.frombuffer(landmarks_raw, dtype=np.float64).reshape(shape)

        if gloss not in gloss_to_id:
            gloss_to_id[gloss] = id
            id_to_gloss[id] = gloss
            id += 1

        yield (gloss_to_id[gloss], landmarks)

def preprocess(label, landmarks):
    landmarks = tf.cast(landmarks, tf.float32)
    landmarks -= tf.reduce_mean(landmarks, axis=1, keepdims=True)  # center
    landmarks /= tf.math.reduce_std(landmarks) + 1e-6  # normalize
    return label, landmarks


output_signature = (tf.TensorSpec(shape=(), dtype = tf.int32),
                    tf.TensorSpec(shape=(41,21,3), dtype = tf.float32))

train_dataset =  tf.data.Dataset.from_generator(generator, output_signature = output_signature)
train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.shuffle(1000).batch(32)
num_classes = len(id_to_gloss)

model = keras.Sequential(
    tf.keras.layers.Input(shape=(41,21,3)),
    tf.keras.layers.Reshape(shape=(41,63)),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GRU(256, return_sequences=False),
    tf.keras.layers.Dense(num_classes, activation = 'softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
