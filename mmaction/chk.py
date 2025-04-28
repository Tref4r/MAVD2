import os

def ck():
    folder_path = "Processed_Videos"
    mp4 = set()
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.mp4'):
                mp4.add(filename)


    folder_path = "Extract_feats/final"
    npy = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.npy'):
                npy.append(filename)

    new_npy = [a.replace("__pose.npy",".mp4") for a in npy]
    n_npy = set(new_npy)

    n_p = list(mp4-n_npy)
    return n_p
    


# ck()
import shutil
def ext_vid():
    n_p = ck()
    os.makedirs('Phase2')
    folder_path = "Processed_Videos"
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.mp4'):
                input_path = os.path.join(root, filename)
                if filename in n_p:
                    shutil.copy(input_path,f"Phase2/{filename}")

# ext_vid()


import os
import shutil

def split_and_move_files(src_folder, dest_folder_base, num_parts):
    # Lấy danh sách tất cả các tệp trong thư mục nguồn
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    # Tính số lượng tệp trong mỗi phần
    num_files = len(files)
    part_size = num_files // num_parts
    
    # Tạo các thư mục đích và di chuyển các tệp
    for i in range(num_parts):
        dest_folder = f"{dest_folder_base}_part{i+1}"
        os.makedirs(dest_folder, exist_ok=True)
        
        # Tính toán phạm vi các tệp cho phần hiện tại
        start_index = i * part_size
        end_index = (i + 1) * part_size if i < num_parts - 1 else num_files
        
        # Di chuyển các tệp vào thư mục đích
        for j in range(start_index, end_index):
            src_file = os.path.join(src_folder, files[j])
            dest_file = os.path.join(dest_folder, files[j])
            shutil.move(src_file, dest_file)

# Sử dụng hàm để chia và di chuyển các tệp
src_folder = "Phase2"
dest_folder_base = "Phase2_split"
num_parts = 5
# split_and_move_files(src_folder, dest_folder_base, num_parts)



def save_filenames_to_txt(folder_path, output_file):
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                f.write(f"{filename}\n")

# Sử dụng hàm để lưu tất cả tên file trong folder vào file txt
folder_path = "Extract_feats/final"
output_file = "filenames.txt"
# save_filenames_to_txt(folder_path, output_file)

import numpy as np
import shutil
def chknpy():
    folder_path = "Extract_feats/final"
    npy = []
    os.makedirs('Feat_kpt')
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.npy'):
                path_np = os.path.join(root, filename)
                new_file = filename.replace("___","__#")
                shutil.copy(path_np, f"Feat_kpt/{new_file}")
                # try:
                #     a=np.load(path_np)
                #     print(a.shape)
                # except:
                #     print('error', path_np)
    return npy
chknpy()