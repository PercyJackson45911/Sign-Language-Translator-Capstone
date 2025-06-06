import os
import json
import shutil

VALID_SPLITS = {'train', 'val', 'test'}
BASE_FOLDER = '/mnt/external/Capstone_Project/Dataset/WSASL/'      # Where you want sorted folders (train, val, test)
UNSORTED_FOLDER = '/mnt/external/Capstone_Project/Dataset/WSASL/videos'  # Your existing folder with all unsorted videos

def main():
    with open('Index.json', 'r') as f:
        data = json.load(f)

    for gloss_entry in data:
        gloss_name = gloss_entry.get('gloss', 'unknown_gloss')
        for instance in gloss_entry.get('instances', []):
            split = instance.get('split')
            if split not in VALID_SPLITS:
                print(f"Skipping unknown split '{split}' for gloss '{gloss_name}'")
                continue

            split_folder = os.path.join(BASE_FOLDER, split)
            os.makedirs(split_folder, exist_ok=True)

            instance_id = instance.get('instance_id', 'noid')
            video_id = instance.get('video_id', 'novideo')
            filename = f"{video_id}.mp4"

            src_path = os.path.join(UNSORTED_FOLDER, filename)
            dst_path = os.path.join(split_folder, filename)

            if not os.path.exists(src_path):
                print(f"Warning: Video file '{filename}' NOT found in unsorted folder. Skipping.")
                continue

            if os.path.exists(dst_path):
                print(f"Already sorted '{filename}', skipping.")
                continue

            print(f"Moving '{filename}' to '{split}' folder.")
            shutil.move(src_path, dst_path)

if __name__ == "__main__":
    main()
