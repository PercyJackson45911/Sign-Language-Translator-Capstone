import os
import cv2
import shutil

def convert_to_25fps(src_path, dst_path, target_fps=25):
    cap = cv2.VideoCapture(src_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return

    size = (frames[0].shape[1], frames[0].shape[0])  # (width, height)

    out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, size)

    for frame in frames:
        out.write(frame)
    out.release()

def process_and_move_videos(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for fname in os.listdir(src_dir):
        if fname.endswith(('.mp4', '.mkv', '.swf')):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, os.path.splitext(fname)[0] + '.mp4')

            convert_to_25fps(src_path, dst_path)
            print(f"Converted: {fname} -> 25 FPS")

def main():
    src = '../Dataset/WSASL/test'
    dst = '../Dataset/WSASL/temp'
    process_and_move_videos(src, dst)

if __name__ == "__main__":
    main()
