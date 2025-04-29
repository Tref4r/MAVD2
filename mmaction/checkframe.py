import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing

def get_video_info(video_path):
    """Lấy số frame và FPS của video"""
    if not os.path.exists(video_path):
        return None, None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count, fps

def convert_video_to_1000_frames(input_path, output_path, target_frame_count=1000):
    """Chuyển video về đúng 1000 frame nếu lớn hơn"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_count <= target_frame_count:
        cap.release()
        return False
    
    frame_indices = np.linspace(0, frame_count - 1, target_frame_count, dtype=int)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    if not out.isOpened():
        cap.release()
        return False
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
    
    cap.release()
    out.release()
    return True

def process_single_video(video_tuple):
    """Hàm xử lý một video - Dùng trong multi-processing"""
    input_path, output_path = video_tuple
    frame_count, fps = get_video_info(input_path)
    
    if frame_count is not None and fps is not None:
        if frame_count > 1000:
            success = convert_video_to_1000_frames(input_path, output_path)
            return f"Processed {input_path}" if success else f"Failed {input_path}"
        else:
            os.system(f"cp '{input_path}' '{output_path}'")  # Copy nếu dưới 1000 frame
            return f"Copied {input_path}"
    return f"Skipped {input_path}"

def process_videos_in_folder(folder_path, output_folder, num_workers=None):
    """Quét thư mục và xử lý video với multi-processing"""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.mp4'):
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, folder_path)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                output_path = os.path.join(output_subfolder, filename)
                video_files.append((input_path, output_path))

    if not video_files:
        print("No videos found!")
        return

    # Multi-processing Pool
    num_workers = num_workers or max(1, 2)
    print(f"Processing {len(video_files)} videos using {num_workers} workers...")
    
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_video, video_files), total=len(video_files)))

    for res in results:
        print(res)

if __name__ == "__main__":
    input_folder = 'temp'
    output_folder = 'Processed_Videos'
    process_videos_in_folder(input_folder, output_folder)
