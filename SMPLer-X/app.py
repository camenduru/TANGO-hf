import os
import shutil
import argparse
import sys
import re
import json
import numpy as np
import os.path as osp
from pathlib import Path
import cv2
import torch
import math
from tqdm import tqdm
from huggingface_hub import hf_hub_download
try:
    import mmpose
except:
    os.system('pip install ./main/transformer_utils')
# hf_hub_download(repo_id="caizhongang/SMPLer-X", filename="smpler_x_h32.pth.tar", local_dir="/home/user/app/pretrained_models")
os.system('cp -rf ./assets/conversions.py /content/myenv/lib/python3.10/site-packages/torchgeometry/core/conversions.py')

def extract_frame_number(file_name):
    match = re.search(r'(\d{5})', file_name)
    if match:
        return int(match.group(1))
    return None

def merge_npz_files(npz_files, output_file):
    npz_files = sorted(npz_files, key=lambda x: extract_frame_number(os.path.basename(x)))
    merged_data = {}
    for file in npz_files:
        data = np.load(file)
        for key in data.files:
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].append(data[key])
    for key in merged_data:
        merged_data[key] = np.stack(merged_data[key], axis=0)
    np.savez(output_file, **merged_data)

def npz_to_npz(pkl_path, npz_path):
    # Load the pickle file
    pkl_example = np.load(pkl_path, allow_pickle=True)
    n = pkl_example["expression"].shape[0]  # Assuming this is the batch size
    full_pose = np.concatenate([pkl_example["global_orient"], pkl_example["body_pose"], pkl_example["jaw_pose"],  pkl_example["leye_pose"], pkl_example["reye_pose"], pkl_example["left_hand_pose"], pkl_example["right_hand_pose"]], axis=1)
    # print(full_pose.shape)
    np.savez(npz_path,
        betas=np.zeros(300),
        poses=full_pose.reshape(n, -1),
        expressions=np.zeros((n, 100)),
        trans=pkl_example["transl"].reshape(n, -1),
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=30,
    )

def get_json(root_dir, output_dir):
    clips = []
    dirs = os.listdir(root_dir)
    all_length = 0
    for dir in dirs:
        if not dir.endswith(".mp4"): continue
        video_id = dir[:-4]
        root = root_dir
        try: 
            length = np.load(os.path.join(root, video_id+".npz"), allow_pickle=True)["poses"].shape[0]
            all_length += length
        except:
            print("cant open ", dir)
            continue
        clip = {
                "video_id": video_id,
                "video_path": root[1:],
                # "audio_path": root,
                "motion_path": root[1:],
                "mode": "test",
                "start_idx": 0,
                "end_idx": length
            }
        clips.append(clip)
    if all_length < 1:
        print(f"skip due to total frames is less than 1500 for {root_dir}")
        return 0 
    else:
        with open(output_dir, 'w') as f:
            json.dump(clips, f, indent=4)
        return all_length


def infer(video_input, in_threshold, num_people, render_mesh, inferer, OUT_FOLDER):
    os.system(f'rm -rf {OUT_FOLDER}/smplx/*')
    multi_person = num_people
    cap = cv2.VideoCapture(video_input)
    video_name = video_input.split("/")[-1]
    success = 1
    frame = 0
    while success:
        success, original_img = cap.read()
        if not success:
            break
        frame += 1
        _, _, _ = inferer.infer(original_img, in_threshold, frame, multi_person, not(render_mesh))
    cap.release()
    npz_files = [os.path.join(OUT_FOLDER, 'smplx', x) for x in os.listdir(os.path.join(OUT_FOLDER, 'smplx'))]
  
    merge_npz_files(npz_files, os.path.join(OUT_FOLDER, video_name.replace(".mp4", ".npz")))
    os.system(f'rm -r {OUT_FOLDER}/smplx')
    npz_to_npz(os.path.join(OUT_FOLDER, video_name.replace(".mp4", ".npz")), os.path.join(OUT_FOLDER, video_name.replace(".mp4", ".npz")))
    source = video_input
    destination = os.path.join(OUT_FOLDER, video_name.replace('.mp4', '.npz')).replace('.npz', '.mp4')
    shutil.copy(source, destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder_path", type=str, default="")
    parser.add_argument("--data_save_path", type=str, default="")
    parser.add_argument("--json_save_path", type=str, default="")
    args = parser.parse_args()
    video_folder = args.video_folder_path

    DEFAULT_MODEL='smpler_x_s32'
    OUT_FOLDER = args.data_save_path
    os.makedirs(OUT_FOLDER, exist_ok=True)
    num_gpus = 1 if torch.cuda.is_available() else -1
    index = torch.cuda.current_device()
    from main.inference import Inferer
    inferer = Inferer(DEFAULT_MODEL, num_gpus, OUT_FOLDER)

    for video_input in tqdm(os.listdir(video_folder)):
        if not video_input.endswith(".mp4"):
            continue
        infer(os.path.join(video_folder, video_input), 0.5, False, False, inferer, OUT_FOLDER)
    get_json(OUT_FOLDER, args.json_save_path)
    