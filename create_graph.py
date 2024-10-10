"""
input: json file with video, audio, motion paths
output: igraph object with nodes containing video, audio, motion, position, velocity, axis_angle, previous, next, frame, fps

preprocess:
1. assume you have a video for one speaker in folder, listed in 
    -- video_a.mp4
    -- video_b.mp4
    run process_video.py to extract frames and audio
"""

import os
import smplx
import torch
import numpy as np
import cv2
import librosa
import igraph
import json
import utils.rotation_conversions as rc
from moviepy.editor import VideoClip, AudioFileClip
from tqdm import tqdm
import imageio
import tempfile
import argparse


def get_motion_reps_tensor(motion_tensor, smplx_model, pose_fps=30, device='cuda'):
    bs, n, _ = motion_tensor.shape
    motion_tensor = motion_tensor.float().to(device)
    motion_tensor_reshaped = motion_tensor.reshape(bs * n, 165)
    
    output = smplx_model(
        betas=torch.zeros(bs * n, 300, device=device),
        transl=torch.zeros(bs * n, 3, device=device),
        expression=torch.zeros(bs * n, 100, device=device),
        jaw_pose=torch.zeros(bs * n, 3, device=device),
        global_orient=torch.zeros(bs * n, 3, device=device),
        body_pose=motion_tensor_reshaped[:, 3:21 * 3 + 3],
        left_hand_pose=motion_tensor_reshaped[:, 25 * 3:40 * 3],
        right_hand_pose=motion_tensor_reshaped[:, 40 * 3:55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(bs * n, 3, device=device),
        reye_pose=torch.zeros(bs * n, 3, device=device),
    )
    
    joints = output['joints'].reshape(bs, n, 127, 3)[:, :, :55, :]
    dt = 1 / pose_fps
    init_vel = (joints[:, 1:2] - joints[:, 0:1]) / dt
    middle_vel = (joints[:, 2:] - joints[:, :-2]) / (2 * dt)
    final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
    vel = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    
    position = joints
    rot_matrices = rc.axis_angle_to_matrix(motion_tensor.reshape(bs, n, 55, 3))
    rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(bs, n, 55, 6)

    init_vel_ang = (motion_tensor[:, 1:2] - motion_tensor[:, 0:1]) / dt
    middle_vel_ang = (motion_tensor[:, 2:] - motion_tensor[:, :-2]) / (2 * dt)
    final_vel_ang = (motion_tensor[:, -1:] - motion_tensor[:, -2:-1]) / dt
    angular_velocity = torch.cat([init_vel_ang, middle_vel_ang, final_vel_ang], dim=1).reshape(bs, n, 55, 3)

    rep15d = torch.cat([position, vel, rot6d, angular_velocity], dim=3).reshape(bs, n, 55 * 15)
    
    return {
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "axis_angle": motion_tensor,
        "angular_velocity": angular_velocity,
        "rep15d": rep15d,
    }



# def get_motion_reps(motion, smplx_model=smplx_model, pose_fps=30):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     smplx_model = smplx.create(
#         "./emage/smplx_models/",
#         model_type='smplx',
#         gender='NEUTRAL_2020',
#         use_face_contour=False,
#         num_betas=300,
#         num_expression_coeffs=100,
#         ext='npz',
#         use_pca=False,
#     ).to(device).eval()
#     print("warning, smplx model is created inside fn for gradio")

#     gt_motion_tensor = motion["poses"]
#     n = gt_motion_tensor.shape[0]
#     bs = 1
#     gt_motion_tensor = torch.from_numpy(gt_motion_tensor).float().to(device).unsqueeze(0)
#     gt_motion_tensor_reshaped = gt_motion_tensor.reshape(bs * n, 165)
#     output = smplx_model(
#         betas=torch.zeros(bs * n, 300).to(device),
#         transl=torch.zeros(bs * n, 3).to(device),
#         expression=torch.zeros(bs * n, 100).to(device),
#         jaw_pose=torch.zeros(bs * n, 3).to(device),
#         global_orient=torch.zeros(bs * n, 3).to(device),
#         body_pose=gt_motion_tensor_reshaped[:, 3:21 * 3 + 3],
#         left_hand_pose=gt_motion_tensor_reshaped[:, 25 * 3:40 * 3],
#         right_hand_pose=gt_motion_tensor_reshaped[:, 40 * 3:55 * 3],
#         return_joints=True,
#         leye_pose=torch.zeros(bs * n, 3).to(device),
#         reye_pose=torch.zeros(bs * n, 3).to(device),
#     )
#     joints = output["joints"].detach().cpu().numpy().reshape(n, 127, 3)[:, :55, :]
#     dt = 1 / pose_fps
#     init_vel = (joints[1:2] - joints[0:1]) / dt
#     middle_vel = (joints[2:] - joints[:-2]) / (2 * dt)
#     final_vel = (joints[-1:] - joints[-2:-1]) / dt
#     vel = np.concatenate([init_vel, middle_vel, final_vel], axis=0)
#     position = joints
#     rot_matrices = rc.axis_angle_to_matrix(gt_motion_tensor.reshape(1, n, 55, 3))[0]
#     rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(n, 55, 6).cpu().numpy()
    
#     init_vel = (motion["poses"][1:2] - motion["poses"][0:1]) / dt
#     middle_vel = (motion["poses"][2:] - motion["poses"][:-2]) / (2 * dt)
#     final_vel = (motion["poses"][-1:] - motion["poses"][-2:-1]) / dt
#     angular_velocity = np.concatenate([init_vel, middle_vel, final_vel], axis=0).reshape(n, 55, 3)

#     rep15d = np.concatenate([
#         position,
#         vel,
#         rot6d,
#         angular_velocity],
#         axis=2
#     ).reshape(n, 55*15)
#     return {
#         "position": position,
#         "velocity": vel,
#         "rotation": rot6d,
#         "axis_angle": motion["poses"],
#         "angular_velocity": angular_velocity,
#         "rep15d": rep15d,
#         "trans": motion["trans"]
#     }

def create_graph(json_path):
    fps = 30
    data_meta = json.load(open(json_path, "r"))
    graph = igraph.Graph(directed=True)
    global_i = 0
    for data_item in data_meta:
        video_path = os.path.join(data_item['video_path'], data_item['video_id'] + ".mp4")
        # audio_path = os.path.join(data_item['audio_path'], data_item['video_id'] +  ".wav")
        motion_path = os.path.join(data_item['motion_path'], data_item['video_id'] +  ".npz")
        video_id = data_item.get("video_id", "")
        motion = np.load(motion_path, allow_pickle=True)
        motion_reps = get_motion_reps(motion)
        position = motion_reps['position']
        velocity = motion_reps['velocity']
        trans = motion_reps['trans']
        axis_angle = motion_reps['axis_angle']
        # audio, sr = librosa.load(audio_path, sr=None)
        # audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        all_frames = []
        reader = imageio.get_reader(video_path)
        all_frames = []
        for frame in reader:
            all_frames.append(frame)
        video_frames = np.array(all_frames)
        min_frames = min(len(video_frames), position.shape[0])
        position = position[:min_frames]
        velocity = velocity[:min_frames]
        video_frames = video_frames[:min_frames]
        # print(min_frames)
        for i in tqdm(range(min_frames)):
            if i == 0:
                previous = -1
                next_node = global_i + 1
            elif i == min_frames - 1:
                previous = global_i - 1
                next_node = -1
            else:
                previous = global_i - 1
                next_node = global_i + 1
            graph.add_vertex(
                idx=global_i,
                name=video_id,
                motion=motion_reps,
                position=position[i],
                velocity=velocity[i],
                axis_angle=axis_angle[i],
                trans=trans[i],
                # audio=audio[],
                video=video_frames[i],
                previous=previous,
                next=next_node,
                frame=i,
                fps=fps,
            )
            global_i += 1
    return graph

def create_edges(graph):
    adaptive_length = [-4, -3, -2, -1, 1, 2, 3, 4]
    # print()
    for i, node in enumerate(graph.vs):
        current_position = node['position']
        current_velocity = node['velocity']
        current_trans = node['trans']
        # print(current_position.shape, current_velocity.shape)
        avg_position = np.zeros(current_position.shape[0])
        avg_velocity = np.zeros(current_position.shape[0])
        avg_trans = 0
        count = 0
        for node_offset in adaptive_length:
            idx = i + node_offset
            if idx < 0 or idx >= len(graph.vs):
                continue
            if node_offset < 0:
              if graph.vs[idx]['next'] == -1:continue
            else:
              if graph.vs[idx]['previous'] == -1:continue
            # add check
            other_node = graph.vs[idx]
            other_position = other_node['position']
            other_velocity = other_node['velocity']
            other_trans = other_node['trans']
            # print(other_position.shape, other_velocity.shape)
            avg_position += np.linalg.norm(current_position - other_position, axis=1)
            avg_velocity += np.linalg.norm(current_velocity - other_velocity, axis=1)
            avg_trans += np.linalg.norm(current_trans - other_trans, axis=0)
            count += 1
        
        if count == 0:
            continue
        threshold_position = avg_position / count
        threshold_velocity = avg_velocity / count
        threshold_trans = avg_trans / count
        # print(threshold_position, threshold_velocity, threshold_trans)
        for j, other_node in enumerate(graph.vs):
            if i == j:
                continue
            if j == node['previous'] or j == node['next']:
                graph.add_edge(i, j, is_continue=1)
                continue
            other_position = other_node['position']
            other_velocity = other_node['velocity']
            other_trans = other_node['trans']
            position_similarity = np.linalg.norm(current_position - other_position, axis=1)
            velocity_similarity = np.linalg.norm(current_velocity - other_velocity, axis=1)
            trans_similarity = np.linalg.norm(current_trans - other_trans, axis=0)
            if trans_similarity < threshold_trans: 
                if np.sum(position_similarity < threshold_position) >= 45 and np.sum(velocity_similarity < threshold_velocity) >= 45:
                    graph.add_edge(i, j, is_continue=0)

    print(f"nodes: {len(graph.vs)}, edges: {len(graph.es)}")
    in_degrees = graph.indegree()
    out_degrees = graph.outdegree()
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    avg_out_degree = sum(out_degrees) / len(out_degrees)
    print(f"Average In-degree: {avg_in_degree}")
    print(f"Average Out-degree: {avg_out_degree}")
    print(f"max in degree: {max(in_degrees)}, max out degree: {max(out_degrees)}")
    print(f"min in degree: {min(in_degrees)}, min out degree: {min(out_degrees)}")
  # igraph.plot(graph, target="/content/test.png", bbox=(1000, 1000), vertex_size=10)
    return graph

def random_walk(graph, walk_length, start_node=None):
    if start_node is None:
        start_node = np.random.choice(graph.vs)
    walk = [start_node]
    is_continue = [1]
    for _ in range(walk_length):
        current_node = walk[-1]
        neighbor_indices = graph.neighbors(current_node.index, mode='OUT')
        if not neighbor_indices:
            break
        next_idx = np.random.choice(neighbor_indices)
        edge_id = graph.get_eid(current_node.index, next_idx)
        is_cont = graph.es[edge_id]['is_continue']
        walk.append(graph.vs[next_idx])
        is_continue.append(is_cont)
    return walk, is_continue


def path_visualization(graph, path, is_continue, save_path, verbose_continue=False, audio_path=None, return_motion=False):
    all_frames = [node['video'] for node in path]
    average_dis_continue = 1 - sum(is_continue) / len(is_continue)
    if verbose_continue:
        print("average_dis_continue:", average_dis_continue)
    duration = len(all_frames) / graph.vs[0]['fps']
    def make_frame(t):
        idx = min(int(t * graph.vs[0]['fps']), len(all_frames) - 1)
        return all_frames[idx]
    video_clip = VideoClip(make_frame, duration=duration)
    if audio_path is not None:
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(save_path, codec='libx264', fps=graph.vs[0]['fps'], audio_codec='aac')

    if return_motion:
        all_motion = [node['axis_angle'] for node in path]
        all_motion = np.stack(all_motion, 0)
        return all_motion

def generate_transition_video(frame_start_path, frame_end_path, output_video_path):
    import subprocess
    import os

    # Define the path to your model and inference script
    model_path = "./frame-interpolation-pytorch/film_net_fp32.pt"
    inference_script = "./frame-interpolation-pytorch/inference.py"

    # Build the command to run the inference script
    command = [
        "python",
        inference_script,
        model_path,
        frame_start_path,
        frame_end_path,
        "--save_path", output_video_path,
        "--gpu",
        "--frames", "3",
        "--fps", "30"
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Generated transition video saved at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while generating transition video: {e}")


def path_visualization_v2(graph, path, is_continue, save_path, verbose_continue=False, audio_path=None, return_motion=False):
    '''
    this is for hugging face demo for fast interpolation. our paper use a diffusion based interpolation method
    '''
    all_frames = [node['video'] for node in path]
    average_dis_continue = 1 - sum(is_continue) / len(is_continue)
    if verbose_continue:
        print("average_dis_continue:", average_dis_continue)
    duration = len(all_frames) / graph.vs[0]['fps']
    
    # First loop: Confirm where blending is needed
    discontinuity_indices = []
    for i, cont in enumerate(is_continue):
        if cont == 0:
            discontinuity_indices.append(i)
    
    # Identify blending positions without overlapping
    blend_positions = []
    processed_frames = set()
    for i in discontinuity_indices:
        # Define the frames for blending: i-2 to i+2
        start_idx = i - 2
        end_idx = i + 2
        # Check index boundaries
        if start_idx < 0 or end_idx >= len(all_frames):
            continue  # Skip if indices are out of bounds
        # Check for overlapping frames
        overlap = any(idx in processed_frames for idx in range(i - 1, i + 2))
        if overlap:
            continue  # Skip if frames have been processed
        # Mark frames as processed
        processed_frames.update(range(i - 1, i + 2))
        blend_positions.append(i)
    
    # Second loop: Perform blending
    temp_dir = tempfile.mkdtemp(prefix='blending_frames_')
    for i in tqdm(blend_positions):
        start_frame_idx = i - 2
        end_frame_idx = i + 2
        frame_start = all_frames[start_frame_idx]
        frame_end = all_frames[end_frame_idx]
        frame_start_path = os.path.join(temp_dir, f'frame_{start_frame_idx}.png')
        frame_end_path = os.path.join(temp_dir, f'frame_{end_frame_idx}.png')
        # Save the start and end frames as images
        imageio.imwrite(frame_start_path, frame_start)
        imageio.imwrite(frame_end_path, frame_end)
        
        # Call FiLM API to generate video
        generated_video_path = os.path.join(temp_dir, f'generated_{start_frame_idx}_{end_frame_idx}.mp4')
        generate_transition_video(frame_start_path, frame_end_path, generated_video_path)
        
        # Read the generated video frames
        reader = imageio.get_reader(generated_video_path)
        generated_frames = [frame for frame in reader]
        reader.close()
        
        # Replace the middle three frames (i-1, i, i+1) in all_frames
        total_generated_frames = len(generated_frames)
        if total_generated_frames < 5:
            print(f"Generated video has insufficient frames ({total_generated_frames}). Skipping blending at position {i}.")
            continue
        middle_start = 1  # Start index for middle 3 frames
        middle_frames = generated_frames[middle_start:middle_start+3]
        for idx, frame_idx in enumerate(range(i - 1, i + 2)):
            all_frames[frame_idx] = middle_frames[idx]
    
    # Create the video clip
    def make_frame(t):
        idx = min(int(t * graph.vs[0]['fps']), len(all_frames) - 1)
        return all_frames[idx]
    
    video_clip = VideoClip(make_frame, duration=duration)
    if audio_path is not None:
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(save_path, codec='libx264', fps=graph.vs[0]['fps'], audio_codec='aac')
    
    if return_motion:
        all_motion = [node['axis_angle'] for node in path]
        all_motion = np.stack(all_motion, 0)
        return all_motion


def graph_pruning(graph):
    ascc = graph.clusters(mode="STRONG")
    lascc = ascc.giant()
    print(f"before nodes: {len(graph.vs)}, edges: {len(graph.es)}")
    print(f"after nodes: {len(lascc.vs)}, edges: {len(lascc.es)}")
    in_degrees = lascc.indegree()
    out_degrees = lascc.outdegree()
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    avg_out_degree = sum(out_degrees) / len(out_degrees)
    print(f"Average In-degree: {avg_in_degree}")
    print(f"Average Out-degree: {avg_out_degree}")
    print(f"max in degree: {max(in_degrees)}, max out degree: {max(out_degrees)}")
    print(f"min in degree: {min(in_degrees)}, min out degree: {min(out_degrees)}")
    return lascc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_save_path", type=str, default="")
    parser.add_argument("--graph_save_path", type=str, default="")
    args = parser.parse_args()
    json_path = args.json_save_path
    graph_path = args.graph_save_path

    # single_test
    # graph = create_graph('/content/drive/MyDrive/003_Codes/TANGO/datasets/data_json/show_oliver_test/Abortion_Laws_-_Last_Week_Tonight_with_John_Oliver_HBO-DRauXXz6t0Y.webm.json')
    graph = create_graph(json_path)
    graph = create_edges(graph)
    # pool_path = "/content/drive/MyDrive/003_Codes/TANGO-JointEmbedding/datasets/oliver_test/show-oliver-test.pkl"
    # graph = igraph.Graph.Read_Pickle(fname=pool_path)
    # graph = igraph.Graph.Read_Pickle(fname="/content/drive/MyDrive/003_Codes/TANGO-JointEmbedding/datasets/oliver_test/test.pkl")
    
    walk, is_continue = random_walk(graph, 100)
    motion = path_visualization(graph, walk, is_continue, "./test.mp4", audio_path=None, verbose_continue=True, return_motion=True)
    # print(motion.shape)
    save_graph = graph.write_pickle(fname=graph_path)
    graph = graph_pruning(graph)

    # show-oliver
    # json_path = "/content/drive/MyDrive/003_Codes/TANGO/datasets/data_json/show_oliver_test/"
    # pre_node_path = "/content/drive/MyDrive/003_Codes/TANGO/datasets/cached_graph/show_oliver_test/"
    # for json_file in tqdm(os.listdir(json_path)):
    #     graph = create_graph(os.path.join(json_path, json_file))
    #     graph = create_edges(graph)
    #     if not len(graph.vs) >= 1500: 
    #         print(f"skip: {len(graph.vs)}", json_file)  
    #     graph.write_pickle(fname=os.path.join(pre_node_path, json_file.split(".")[0] + ".pkl"))
    #     print(f"Graph saved at {json_file.split('.')[0]}.pkl")