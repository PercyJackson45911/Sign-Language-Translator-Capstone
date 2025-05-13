import os
import json
import shutil

# Path to dataset and Index.json
video_path = '../Dataset/WSASL/videos'  # Folder containing your video files
index_json_path = '../Dataset/WSASL/Index.json'  # Path to the Index.json file

# Paths for split directories
train_dir = '../Dataset/WSASL/train'
test_dir = '../Dataset/WSASL/test'
val_dir = '../Dataset/WSASL/val'

# Load Index.json
with open(index_json_path, 'r') as f:
    data = json.load(f)

# Ensure target folders exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to move videos based on split info
def move_video(video_id, split):
    video_file = os.path.join(video_path, f'{video_id}.mp4')

    # Check if the video exists
    if os.path.exists(video_file):
        # Determine the destination based on split value
        if split == 'train':
            dest_dir = train_dir
        elif split == 'test':
            dest_dir = test_dir
        elif split == 'val':
            dest_dir = val_dir
        else:
            print(f"Unknown split: {split}")
            return

        # Move the video to the correct folder
        shutil.move(video_file, os.path.join(dest_dir, f'{video_id}.mp4'))
        print(f"Moved {video_id} to {split} folder.")
    else:
        print(f"Video {video_id} not found. Removing from Index.json.")
        return video_id

# Process the data and move the videos
videos_to_remove = []
for x in data:
    for instance in x['instances']:
        video_id = instance['video_id']
        split = instance['split']
        failed_video = move_video(video_id, split)

        if failed_video:
            videos_to_remove.append(failed_video)

# Remove the failed videos from Index.json
for x in data:
    x['instances'] = [instance for instance in x['instances'] if instance['video_id'] not in videos_to_remove]

# Save the updated Index.json
with open(index_json_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Index.json updated. Failed videos removed.")
