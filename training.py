import tensorflow as tf
import numpy as np

# Load dataset
datasets = tf.data.TFRecordDataset('../Dataset/tfrecords/train.tfrecord')

# Take one record
raw_record = next(iter(datasets))

# Parse the record
example = tf.train.Example()
example.ParseFromString(raw_record.numpy())

# Extract features
result = {}
for key, feature in example.features.feature.items():
    kind = feature.WhichOneof('kind')  # Correct usage
    if kind:
        value = getattr(feature, kind).value
        if key == 'video_id':
            result['video_id'] = value[0].decode('utf-8')
        elif key == 'landmarks':
            result['landmarks'] = np.frombuffer(value[0], dtype=np.float64)  # Use correct dtype
        elif key == 'shape':
            result['shape'] = tuple(feature.int64_list.value)

# Optional debug prints
print("Raw shape stored:", result.get('shape'))
print("Landmarks size:", result.get('landmarks').size if 'landmarks' in result else 'missing')

# Reshape if possible
if 'landmarks' in result and 'shape' in result:
    expected_size = np.prod(result['shape'])
