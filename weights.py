import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json

train_filepath = '/mnt/external/Capstone_Project/Dataset/pt_files/train'
test_filepath = '/mnt/external/Capstone_Project/Dataset/pt_files/test'
val_filepath = '/mnt/external/Capstone_Project/Dataset/pt_files/val'

with open('Index.json', 'r') as f:
    data = json.load(f)

class DatasetMaker6969Mrk7432(Dataset):
    def __init__(self, data, root_filepath, split):
        self.data = data  # your JSON loaded data
        self.root_filepath = root_filepath
        self.samples = []
        self.failed_samples = []
        self.split = split
        for entry in data:
            gloss = entry['gloss']
            for instance in entry['instance']:
                if instance['split'] == split:
                    video_id = instance['video_id']
                else:
                    continue
                filepath = os.path.join(root_filepath, split, video_id)
                if os.path.exists(filepath):
                    self.samples.append((filepath, gloss))
                else:
                    self.failed_samples.append(video_id)

        # Build gloss_to_id for label encoding
        unique_glosses = set([s[1] for s in self.samples])
        self.gloss_to_id = {g: i for i, g in enumerate(sorted(unique_glosses))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, gloss = self.samples[idx]
        pt = torch.load(filepath)
        landmarks = pt['landmarks']  # Tensor list or tensor, confirm this
        # Pad landmarks to longest seq in batch later, so just return raw here
        label = self.gloss_to_id[gloss]
        return landmarks, label


def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, labels

train_dataset = DatasetMaker6969Mrk7432(data, train_filepath, 'train')
