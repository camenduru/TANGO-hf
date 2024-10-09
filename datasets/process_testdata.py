# import smplx
# import torch
# import pickle
# import numpy as np

# # Global: Load the SMPL-X model once
# smplx_model = smplx.create(
#     "/content/drive/MyDrive/003_Codes/TANGO-JointEmbedding/emage/smplx_models/", 
#     model_type='smplx',
#     gender='NEUTRAL_2020', 
#     use_face_contour=False,
#     num_betas=300,
#     num_expression_coeffs=100, 
#     ext='npz',
#     use_pca=True,
#     num_pca_comps=12,
# ).to("cuda").eval()

# device = "cuda"

# def pkl_to_npz(pkl_path, npz_path):
#     # Load the pickle file
#     with open(pkl_path, "rb") as f:
#         pkl_example = pickle.load(f)

#     bs = 1
#     n = pkl_example["expression"].shape[0]  # Assuming this is the batch size

#     # Convert numpy arrays to torch tensors
#     def to_tensor(numpy_array):
#         return torch.tensor(numpy_array, dtype=torch.float32).to(device)

#     # Ensure that betas are loaded from the pickle data, converting them to torch tensors
#     betas = to_tensor(pkl_example["betas"])
#     transl = to_tensor(pkl_example["transl"])
#     expression = to_tensor(pkl_example["expression"])
#     jaw_pose = to_tensor(pkl_example["jaw_pose"])
#     global_orient = to_tensor(pkl_example["global_orient"])
#     body_pose_axis = to_tensor(pkl_example["body_pose_axis"])
#     left_hand_pose = to_tensor(pkl_example['left_hand_pose'])
#     right_hand_pose = to_tensor(pkl_example['right_hand_pose'])
#     leye_pose = to_tensor(pkl_example['leye_pose'])
#     reye_pose = to_tensor(pkl_example['reye_pose'])

#     # Pass the loaded data into the SMPL-X model
#     gt_vertex = smplx_model(
#         betas=betas,
#         transl=transl,  # Translation
#         expression=expression,  # Expression
#         jaw_pose=jaw_pose,  # Jaw pose
#         global_orient=global_orient,  # Global orientation
#         body_pose=body_pose_axis,  # Body pose
#         left_hand_pose=left_hand_pose,  # Left hand pose
#         right_hand_pose=right_hand_pose,  # Right hand pose
#         return_full_pose=True,
#         leye_pose=leye_pose,  # Left eye pose
#         reye_pose=reye_pose,  # Right eye pose
#     )

#     # Save the relevant data to an npz file
#     np.savez(npz_path,
#         betas=pkl_example["betas"],
#         poses=gt_vertex["full_pose"].cpu().numpy(),
#         expressions=pkl_example["expression"],
#         trans=pkl_example["transl"],
#         model='smplx2020',
#         gender='neutral',
#         mocap_frame_rate=30,
#     )

# from tqdm import tqdm
# import os
# def convert_all_pkl_in_folder(folder_path):
#     # Collect all .pkl files
#     pkl_files = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".pkl"):
#                 pkl_files.append(os.path.join(root, file))
    
#     # Process each file with a progress bar
#     for pkl_path in tqdm(pkl_files, desc="Converting .pkl to .npz"):
#         npz_path = pkl_path.replace(".pkl", ".npz")  # Replace .pkl with .npz
#         pkl_to_npz(pkl_path, npz_path)

# convert_all_pkl_in_folder("/content/oliver/oliver/")  


# import os
# import json

# def collect_dataset_info(root_dir):
#     dataset_info = []
    
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith(".npz"):
#                 video_id = file[:-4]  # Removing the .npz extension to get the video ID

#                 # Construct the paths based on the current root directory
#                 motion_path = os.path.join(root)
#                 video_path = os.path.join(root)
#                 audio_path = os.path.join(root)
                
#                 # Determine the mode (train, val, test) by checking parent directory
#                 mode = root.split(os.sep)[-2]  # Assuming mode is one folder up in hierarchy

#                 dataset_info.append({
#                     "video_id": video_id,
#                     "video_path": video_path,
#                     "audio_path": audio_path,
#                     "motion_path": motion_path,
#                     "mode": mode  
#                 })
#     return dataset_info

# # Set the root directory path of your dataset
# root_dir = '/content/oliver/oliver/'  # Adjust this to your actual root directory
# dataset_info = collect_dataset_info(root_dir)
# output_file = '/content/drive/MyDrive/003_Codes/TANGO-JointEmbedding/datasets/show-oliver-original.json'

# # Save the dataset information to a JSON file
# with open(output_file, 'w') as json_file:
#     json.dump(dataset_info, json_file, indent=4)
# print(f"Dataset information saved to {output_file}")


# import os
# import json
# import numpy as np

# def load_npz(npz_path):
#     try:
#         data = np.load(npz_path)
#         return data
#     except Exception as e:
#         print(f"Error loading {npz_path}: {e}")
#         return None

# def generate_clips(data, stride, window_length):
#     clips = []
#     for entry in data:
#         npz_data = load_npz(os.path.join(entry['motion_path'],entry['video_id']+".npz"))

#         # Only continue if the npz file is successfully loaded
#         if npz_data is None:
#             continue

#         # Determine the total length of the sequence from npz data
#         total_frames = npz_data["poses"].shape[0]

#         # Generate clips based on stride and window_length
#         for start_idx in range(0, total_frames - window_length + 1, stride):
#             end_idx = start_idx + window_length
#             clip = {
#                 "video_id": entry["video_id"],
#                 "video_path": entry["video_path"],
#                 "audio_path": entry["audio_path"],
#                 "motion_path": entry["motion_path"],
#                 "mode": entry["mode"],
#                 "start_idx": start_idx,
#                 "end_idx": end_idx
#             }
#             clips.append(clip)

#     return clips

# # Load the existing dataset JSON file
# input_json = '/content/drive/MyDrive/003_Codes/TANGO-JointEmbedding/datasets/show-oliver-original.json'
# with open(input_json, 'r') as f:
#     dataset_info = json.load(f)

# # Set stride and window length
# stride = 40  # Adjust stride as needed
# window_length = 64  # Adjust window length as needed

# # Generate clips for all data
# clips_data = generate_clips(dataset_info, stride, window_length)

# # Save the filtered clips data to a new JSON file
# output_json = f'/content/drive/MyDrive/003_Codes/TANGO-JointEmbedding/datasets/show-oliver-s{stride}_w{window_length}.json'
# with open(output_json, 'w') as f:
#     json.dump(clips_data, f, indent=4)

# print(f"Filtered clips data saved to {output_json}")



from ast import Expression
import os
import numpy as np
import wave
from moviepy.editor import VideoFileClip

def split_npz(npz_path, output_prefix):
    try:
        # Load the npz file
        data = np.load(npz_path)

        # Get the arrays and split them along the time dimension (T)
        poses = data["poses"]
        betas = data["betas"]
        expressions = data["expressions"]
        trans = data["trans"]

        # Determine the halfway point (T/2)
        half = poses.shape[0] // 2

        # Save the first half (0-5 seconds)
        np.savez(output_prefix + "_0_5.npz",
                 betas=betas[:half],
                 poses=poses[:half],
                 expressions=expressions[:half],
                 trans=trans[:half],
                 model=data['model'],
                 gender=data['gender'],
                 mocap_frame_rate=data['mocap_frame_rate'])

        # Save the second half (5-10 seconds)
        np.savez(output_prefix + "_5_10.npz",
                 betas=betas[half:],
                 poses=poses[half:],
                 expressions=expressions[half:],
                 trans=trans[half:],
                 model=data['model'],
                 gender=data['gender'],
                 mocap_frame_rate=data['mocap_frame_rate'])

        print(f"NPZ split saved for {output_prefix}")
    except Exception as e:
        print(f"Error processing NPZ file {npz_path}: {e}")

def split_wav(wav_path, output_prefix):
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(wav_file.getnframes())
            half_frame = len(frames) // 2

            # Create two half files
            for i, start_frame in enumerate([0, half_frame]):
                with wave.open(f"{output_prefix}_{i*5}_{(i+1)*5}.wav", 'wb') as out_wav:
                    out_wav.setparams(params)
                    if i == 0:
                        out_wav.writeframes(frames[:half_frame])
                    else:
                        out_wav.writeframes(frames[half_frame:])
        print(f"WAV split saved for {output_prefix}")
    except Exception as e:
        print(f"Error processing WAV file {wav_path}: {e}")

def split_mp4(mp4_path, output_prefix):
    try:
        clip = VideoFileClip(mp4_path)
        for i in range(2):
            subclip = clip.subclip(i*5, (i+1)*5)
            subclip.write_videofile(f"{output_prefix}_{i*5}_{(i+1)*5}.mp4", codec="libx264", audio_codec="aac")
        print(f"MP4 split saved for {output_prefix}")
    except Exception as e:
        print(f"Error processing MP4 file {mp4_path}: {e}")

def process_files(root_dir, output_dir):
    import json
    clips = []
    dirs = os.listdir(root_dir)
    for dir in dirs:
        video_id = dir
        output_prefix = os.path.join(output_dir, video_id)
        root = os.path.join(root_dir, dir)
        npz_path = os.path.join(root, video_id + ".npz")
        wav_path = os.path.join(root, video_id + ".wav")
        mp4_path = os.path.join(root, video_id + ".mp4")

        # split_npz(npz_path, output_prefix)
        # split_wav(wav_path, output_prefix)
        # split_mp4(mp4_path, output_prefix)

        clip = {
                "video_id": video_id,
                "video_path": root,
                "audio_path": root,
                "motion_path": root,
                "mode": "test",
                "start_idx": 0,
                "end_idx": 150
            }
        clips.append(clip)

    output_json = output_dir + "/test.json"
    with open(output_json, 'w') as f:
        json.dump(clips, f, indent=4)
    

# Set the root directory path of your dataset and output directory
root_dir = '/content/oliver/oliver/Abortion_Laws_-_Last_Week_Tonight_with_John_Oliver_HBO-DRauXXz6t0Y.webm/test/'
output_dir = '/content/test'

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all the files
process_files(root_dir, output_dir)
