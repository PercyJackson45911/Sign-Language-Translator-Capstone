import tensorflow as tf
import numpy as np

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

output_signature = (tf.TensorSpec(shape=(), dtype = tf.int32),
                    tf.TensorSpec(shape=(41,21,3), dtype = tf.float32))

dataset =  tf.data.Dataset.from_generator(generator, output_signature = output_signature)
