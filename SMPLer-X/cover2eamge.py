import smplx
import torch
import pickle
import numpy as np

# # Global: Load the SMPL-X model once
# smplx_model = smplx.create(
#     "/content/drive/MyDrive/003_Codes/TANGO-JointEmbedding/beat2/smplx_models/", 
#     model_type='smplx',
#     gender='NEUTRAL_2020', 
#     use_face_contour=False,
#     num_betas=10,
#     num_expression_coeffs=10, 
#     ext='npz',
#     use_pca=False,
# ).to("cuda").eval()

# device = "cuda"

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

# smplierx data
def npz_to_npz_v2(pkl_path, npz_path):
    # Load the pickle file
    pkl_example = np.load(pkl_path, allow_pickle=True)

    bs = 1
    n = pkl_example["expression"].shape[0]  # Assuming this is the batch size

    # Convert numpy arrays to torch tensors
    def to_tensor(numpy_array):
        return torch.tensor(numpy_array, dtype=torch.float32).to(device)

    # Ensure that betas are loaded from the pickle data, converting them to torch tensors
    betas = to_tensor(pkl_example["betas"]).reshape(n, -1)
    transl = to_tensor(pkl_example["transl"]).reshape(n, -1)
    expression = to_tensor(pkl_example["expression"]).reshape(n, -1)
    jaw_pose = to_tensor(pkl_example["jaw_pose"]).reshape(n, -1)
    global_orient = to_tensor(pkl_example["global_orient"]).reshape(n, -1)
    body_pose_axis = to_tensor(pkl_example["body_pose"]).reshape(n, -1)
    left_hand_pose = to_tensor(pkl_example['left_hand_pose']).reshape(n, -1)
    right_hand_pose = to_tensor(pkl_example['right_hand_pose']).reshape(n, -1)
    leye_pose = to_tensor(pkl_example['leye_pose']).reshape(n, -1)
    reye_pose = to_tensor(pkl_example['reye_pose']).reshape(n, -1)

    # print(left_hand_pose.shape, right_hand_pose.shape)

    # Pass the loaded data into the SMPL-X model
    gt_vertex = smplx_model(
        betas=betas,
        transl=transl,  # Translation
        expression=expression,  # Expression
        jaw_pose=jaw_pose,  # Jaw pose
        global_orient=global_orient,  # Global orientation
        body_pose=body_pose_axis,  # Body pose
        left_hand_pose=left_hand_pose,  # Left hand pose
        right_hand_pose=right_hand_pose,  # Right hand pose
        return_full_pose=True,
        leye_pose=leye_pose,  # Left eye pose
        reye_pose=reye_pose,  # Right eye pose
    )

    # Save the relevant data to an npz file
    np.savez(npz_path,
        betas=np.zeros((n, 300)),
        poses=gt_vertex["full_pose"].cpu().numpy(),
        expressions=np.zeros((n, 100)),
        trans=pkl_example["transl"].reshape(n, -1),
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=30,
    )

# smplierx data
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
        trans=np.zeros((n, 3)),
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=30,
    )
if __name__ == "__main__":
    npz_to_npz("/content/drive/MyDrive/003_Codes/TANGO/SMPLer-X/demo/outputs/results_smplx.npz", "/content/drive/MyDrive/003_Codes/TANGO/SMPLer-X/demo/outputs/results_smplx_emage.npz")