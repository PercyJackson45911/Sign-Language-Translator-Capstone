import json
import os

json_path = '../Dataset/WSASL/Index.json'
video_dir = '../Dataset/WSASL/val'

with open(json_path, 'r') as f:
    data = json.load(f)

for entry in data:
    entry['instances'] = [
        inst for inst in entry['instances']
        if not (
            inst['split'] == 'val' and
            not os.path.isfile(os.path.join(video_dir, f"{inst['video_id']}.mp4"))
        )
    ]

# Save updated data
with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Done cleaning missing train videos.")
