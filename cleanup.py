import os
import json

# Path to the index file
index_path = '../Dataset/WSASL/Index.json'

# Load the index
with open(index_path, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} gloss entries from Index.json")

valid_entries = []

# Check each gloss entry
for entry in data:
    gloss = entry.get('gloss', '')
    instances = entry.get('instances', [])
    new_instances = []

    for inst in instances:
        split = inst.get('split')
        video_id = inst.get('video_id')
        video_path = f'../Dataset/WSASL/{split}/{video_id}.mp4'

        if os.path.isfile(video_path):
            new_instances.append(inst)
        else:
            print(f"Missing file: {video_path} — removing instance")

    # Only keep entries that still have valid instances
    if new_instances:
        entry['instances'] = new_instances
        valid_entries.append(entry)
    else:
        print(f"All instances missing for gloss '{gloss}', removing entry")

# Save the cleaned index
cleaned_index_path = '../Dataset/WSASL/Index_cleaned.json'
with open(cleaned_index_path, 'w') as f:
    json.dump(valid_entries, f, indent=4)

print(f"\nCleaned index saved to {cleaned_index_path} with {len(valid_entries)} valid gloss entries")
