import spaces
import os
# os.environ["XDG_RUNTIME_DIR"] = "/content"
# os.system("Xvfb :99 -ac &")
# os.environ["DISPLAY"] = ":99"
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
import gradio as gr
import gc
import soundfile as sf
import shutil
import argparse
from moviepy.tools import verbose_print
from omegaconf import OmegaConf
import random
import numpy as np
import json 
import librosa
import emage.mertic
from datetime import datetime
from decord import VideoReader
from PIL import Image
import copy

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import smplx
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import igraph

# import emage
import utils.rotation_conversions as rc
from utils.video_io import save_videos_from_pil
from utils.genextend_inference_utils import adjust_statistics_to_match_reference
from create_graph import path_visualization, graph_pruning, get_motion_reps_tensor, path_visualization_v2


def search_path_dp(graph, audio_low_np, audio_high_np, loop_penalty=0.1, top_k=1, search_mode="both", continue_penalty=0.1):
    T = audio_low_np.shape[0]  # Total time steps
    N = len(graph.vs)          # Total number of nodes in the graph

    # Initialize DP tables
    min_cost = [{} for _ in range(T)]  # min_cost[t][node_index] = list of tuples: (cost, prev_node_index, prev_tuple_index, non_continue_count, visited_nodes)

    # Initialize the first time step
    start_nodes = [v for v in graph.vs if v['previous'] is None or v['previous'] == -1]
    for node in start_nodes:
        node_index = node.index
        motion_low = node['motion_low']      # Shape: [C]
        motion_high = node['motion_high']    # Shape: [C]

        # Cost using cosine similarity
        if search_mode == "both":
            cost = 2 - (np.dot(audio_low_np[0], motion_low.T) + np.dot(audio_high_np[0], motion_high.T))
        elif search_mode == "high_level":
            cost = 1 - np.dot(audio_high_np[0], motion_high.T)
        elif search_mode == "low_level":
            cost = 1 - np.dot(audio_low_np[0], motion_low.T)

        visited_nodes = {node_index: 1}  # Initialize visit count as a dictionary

        min_cost[0][node_index] = [ (cost, None, None, 0, visited_nodes) ]  # Initialize with no predecessor and 0 non-continue count

    # DP over time steps
    for t in range(1, T):
        for node in graph.vs:
            node_index = node.index
            candidates = []

            # Incoming edges to the current node
            incoming_edges = graph.es.select(_to=node_index)
            for edge in incoming_edges:
                prev_node_index = edge.source
                edge_id = edge.index
                is_continue_edge = graph.es[edge_id]['is_continue']
                prev_node = graph.vs[prev_node_index]
                if prev_node_index in min_cost[t-1]:
                    for tuple_index, (prev_cost, _, _, prev_non_continue_count, prev_visited) in enumerate(min_cost[t-1][prev_node_index]):
                        # Loop punishment
                        if node_index in prev_visited:
                            loop_time = prev_visited[node_index]  # Get the count of previous visits
                            loop_cost = prev_cost + loop_penalty * np.exp(loop_time)  # Apply exponential penalty
                            new_visited = prev_visited.copy()
                            new_visited[node_index] = loop_time + 1  # Increment visit count
                        else:
                            loop_cost = prev_cost
                            new_visited = prev_visited.copy()
                            new_visited[node_index] = 1  # Initialize visit count for the new node

                        motion_low = node['motion_low']      # Shape: [C]
                        motion_high = node['motion_high']    # Shape: [C]

                        if search_mode == "both":
                            cost_increment = 2 - (np.dot(audio_low_np[t], motion_low.T) + np.dot(audio_high_np[t], motion_high.T))
                        elif search_mode == "high_level":
                            cost_increment = 1 - np.dot(audio_high_np[t], motion_high.T)
                        elif search_mode == "low_level":
                            cost_increment = 1 - np.dot(audio_low_np[t], motion_low.T)

                        # Check if the edge is "is_continue"
                        if not is_continue_edge:
                            non_continue_count = prev_non_continue_count + 1  # Increment the count of non-continue edges
                        else:
                            non_continue_count = prev_non_continue_count

                        # Apply the penalty based on the square of the number of non-continuous edges
                        continue_penalty_cost = continue_penalty * non_continue_count

                        total_cost = loop_cost + cost_increment + continue_penalty_cost

                        candidates.append( (total_cost, prev_node_index, tuple_index, non_continue_count, new_visited) )

            # Keep the top k candidates
            if candidates:
                # Sort candidates by total_cost
                candidates.sort(key=lambda x: x[0])
                # Keep top k
                min_cost[t][node_index] = candidates[:top_k]
            else:
                # No candidates, do nothing
                pass

    # Collect all possible end paths at time T-1
    end_candidates = []
    for node_index, tuples in min_cost[T-1].items():
        for tuple_index, (cost, _, _, _, _) in enumerate(tuples):
            end_candidates.append( (cost, node_index, tuple_index) )

    if not end_candidates:
        print("No valid path found.")
        return [], []

    # Sort end candidates by cost
    end_candidates.sort(key=lambda x: x[0])

    # Keep top k paths
    top_k_paths_info = end_candidates[:top_k]

    # Reconstruct the paths
    optimal_paths = []
    is_continue_lists = []
    for final_cost, node_index, tuple_index in top_k_paths_info:
        optimal_path_indices = []
        current_node_index = node_index
        current_tuple_index = tuple_index
        for t in range(T-1, -1, -1):
            optimal_path_indices.append(current_node_index)
            tuple_data = min_cost[t][current_node_index][current_tuple_index]
            _, prev_node_index, prev_tuple_index, _, _ = tuple_data
            current_node_index = prev_node_index
            current_tuple_index = prev_tuple_index
            if current_node_index is None:
                break  # Reached the start node
        optimal_path_indices = optimal_path_indices[::-1]  # Reverse to get correct order
        optimal_path = [graph.vs[idx] for idx in optimal_path_indices]
        optimal_paths.append(optimal_path)

        # Extract continuity information
        is_continue = []
        for i in range(len(optimal_path) - 1):
            edge_id = graph.get_eid(optimal_path[i].index, optimal_path[i + 1].index)
            is_cont = graph.es[edge_id]['is_continue']
            is_continue.append(is_cont)
        is_continue_lists.append(is_continue)

    print("Top {} Paths:".format(len(optimal_paths)))
    for i, path in enumerate(optimal_paths):
        path_indices = [node.index for node in path]
        print("Path {}: Cost: {}, Nodes: {}".format(i+1, top_k_paths_info[i][0], path_indices))

    return optimal_paths, is_continue_lists


def test_fn(model, device, iteration, candidate_json_path, test_path, cfg, audio_path, **kwargs):
    torch.set_grad_enabled(False)
    pool_path = candidate_json_path.replace("data_json", "cached_graph").replace(".json", ".pkl")
    graph = igraph.Graph.Read_Pickle(fname=pool_path)
    # print(len(graph.vs))

    save_dir = os.path.join(test_path, f"retrieved_motions_{iteration}")
    os.makedirs(save_dir, exist_ok=True)

    actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    actual_model.eval()

    # with open(candidate_json_path, 'r') as f:
    #     candidate_data = json.load(f)
    all_motions = {}
    for i, node in enumerate(graph.vs):
        if all_motions.get(node["name"]) is None:
            all_motions[node["name"]] = [node["axis_angle"].reshape(-1)]
        else:
            all_motions[node["name"]].append(node["axis_angle"].reshape(-1))
    for k, v in all_motions.items():
        all_motions[k] = np.stack(v) # T, J*3
        # print(k, all_motions[k].shape)
    
    window_size = cfg.data.pose_length
    motion_high_all = []
    motion_low_all = []
    for k, v in all_motions.items():
        motion_tensor = torch.from_numpy(v).float().to(device).unsqueeze(0)
        _, t, _ = motion_tensor.shape
        
        if t >= window_size:
            num_chunks = t // window_size
            motion_high_list = []
            motion_low_list = []

            for i in range(num_chunks):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                motion_slice = motion_tensor[:, start_idx:end_idx, :]
                
                motion_features = actual_model.get_motion_features(motion_slice)
                
                motion_low = motion_features["motion_low"].cpu().numpy()
                motion_high = motion_features["motion_cls"].unsqueeze(0).repeat(1, motion_low.shape[1], 1).cpu().numpy()

                motion_high_list.append(motion_high[0])
                motion_low_list.append(motion_low[0])

            remain_length = t % window_size
            if remain_length > 0:
                start_idx = t - window_size
                motion_slice = motion_tensor[:, start_idx:, :]

                motion_features = actual_model.get_motion_features(motion_slice)
                # motion_high = motion_features["motion_high_weight"].cpu().numpy()
                motion_low = motion_features["motion_low"].cpu().numpy()
                motion_high = motion_features["motion_cls"].unsqueeze(0).repeat(1, motion_low.shape[1], 1).cpu().numpy()

                motion_high_list.append(motion_high[0][-remain_length:])
                motion_low_list.append(motion_low[0][-remain_length:])

            motion_high_all.append(np.concatenate(motion_high_list, axis=0))
            motion_low_all.append(np.concatenate(motion_low_list, axis=0))

        else: # t < window_size:
            gap = window_size - t
            motion_slice = torch.cat([motion_tensor, torch.zeros((motion_tensor.shape[0], gap, motion_tensor.shape[2])).to(motion_tensor.device)], 1)
            motion_features = actual_model.get_motion_features(motion_slice)
            # motion_high = motion_features["motion_high_weight"].cpu().numpy()
            motion_low = motion_features["motion_low"].cpu().numpy()
            motion_high = motion_features["motion_cls"].unsqueeze(0).repeat(1, motion_low.shape[1], 1).cpu().numpy()

            motion_high_all.append(motion_high[0][:t])
            motion_low_all.append(motion_low[0][:t])
            
    motion_high_all = np.concatenate(motion_high_all, axis=0)
    motion_low_all = np.concatenate(motion_low_all, axis=0)
    # print(motion_high_all.shape, motion_low_all.shape, len(graph.vs))
    motion_low_all = motion_low_all / np.linalg.norm(motion_low_all, axis=1, keepdims=True)
    motion_high_all = motion_high_all / np.linalg.norm(motion_high_all, axis=1, keepdims=True)
    assert motion_high_all.shape[0] == len(graph.vs)
    assert motion_low_all.shape[0] == len(graph.vs)
    
    for i, node in enumerate(graph.vs):
        node["motion_high"] = motion_high_all[i]
        node["motion_low"] = motion_low_all[i]

    graph = graph_pruning(graph)
    # for gradio, use a subgraph
    if len(graph.vs) > 1800:
        gap = len(graph.vs) - 1800
        start_d = random.randint(0, 1800)
        graph.delete_vertices(range(start_d, start_d + gap))
    ascc_2 = graph.clusters(mode="STRONG")
    graph = ascc_2.giant()

    # drop the id of gt
    idx = 0
    audio_waveform, sr = librosa.load(audio_path)
    audio_waveform = librosa.resample(audio_waveform, orig_sr=sr, target_sr=cfg.data.audio_sr)
    audio_tensor = torch.from_numpy(audio_waveform).float().to(device).unsqueeze(0)
    
    target_length = audio_tensor.shape[1] // cfg.data.audio_sr * 30
    window_size = int(cfg.data.audio_sr * (cfg.data.pose_length / 30))
    _, t = audio_tensor.shape
    audio_low_list = []
    audio_high_list = []

    if t >= window_size:
        num_chunks = t // window_size
        # print(num_chunks, t % window_size)
        for i in range(num_chunks):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            # print(start_idx, end_idx, window_size)
            audio_slice = audio_tensor[:, start_idx:end_idx]

            model_out_candidates = actual_model.get_audio_features(audio_slice)
            audio_low = model_out_candidates["audio_low"]
            # audio_high = model_out_candidates["audio_high_weight"]
            audio_high = model_out_candidates["audio_cls"].unsqueeze(0).repeat(1, audio_low.shape[1], 1)
            # print(audio_low.shape, audio_high.shape)

            audio_low = F.normalize(audio_low, dim=2)[0].cpu().numpy()
            audio_high = F.normalize(audio_high, dim=2)[0].cpu().numpy()

            audio_low_list.append(audio_low)
            audio_high_list.append(audio_high)
            # print(audio_low.shape, audio_high.shape)
            

        remain_length = t % window_size
        if remain_length > 1:
            start_idx = t - window_size
            audio_slice = audio_tensor[:, start_idx:]

            model_out_candidates = actual_model.get_audio_features(audio_slice)
            audio_low = model_out_candidates["audio_low"]
            # audio_high = model_out_candidates["audio_high_weight"]
            audio_high = model_out_candidates["audio_cls"].unsqueeze(0).repeat(1, audio_low.shape[1], 1)
            
            gap = target_length - np.concatenate(audio_low_list, axis=0).shape[1]
            audio_low = F.normalize(audio_low, dim=2)[0][-gap:].cpu().numpy()
            audio_high = F.normalize(audio_high, dim=2)[0][-gap:].cpu().numpy()
            
            # print(audio_low.shape, audio_high.shape)
            audio_low_list.append(audio_low)
            audio_high_list.append(audio_high)
    else:
        gap = window_size - t
        audio_slice = audio_tensor 
        model_out_candidates = actual_model.get_audio_features(audio_slice)
        audio_low = model_out_candidates["audio_low"]
        # audio_high = model_out_candidates["audio_high_weight"]
        audio_high = model_out_candidates["audio_cls"].unsqueeze(0).repeat(1, audio_low.shape[1], 1)
            
        gap = target_length - np.concatenate(audio_low_list, axis=0).shape[1]
        audio_low = F.normalize(audio_low, dim=2)[0][:gap].cpu().numpy()
        audio_high = F.normalize(audio_high, dim=2)[0][:gap].cpu().numpy()
        audio_low_list.append(audio_low)
        audio_high_list.append(audio_high)
    
    audio_low_all = np.concatenate(audio_low_list, axis=0)
    audio_high_all = np.concatenate(audio_high_list, axis=0)
    path_list, is_continue_list = search_path_dp(graph, audio_low_all, audio_high_all, top_k=1, search_mode="both")
    
    res_motion = []
    counter = 0
    for path, is_continue in zip(path_list, is_continue_list):
        # print(path)
        # res_motion_current = path_visualization(
        #   graph, path, is_continue, os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4"), audio_path=audio_path, return_motion=True, verbose_continue=True
        # )
        res_motion_current = path_visualization_v2(
          graph, path, is_continue, os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4"), audio_path=audio_path, return_motion=True, verbose_continue=True
        )

        video_temp_path = os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4")
        
        video_reader = VideoReader(video_temp_path)
        video_np = []
        for i in range(len(video_reader)):
            if i == 0: continue
            video_frame = video_reader[i].asnumpy()
            video_np.append(Image.fromarray(video_frame))
        adjusted_video_pil = adjust_statistics_to_match_reference([video_np])
        save_videos_from_pil(adjusted_video_pil[0], os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4"), fps=30, bitrate=2000000)


        audio_temp_path = audio_path
        lipsync_output_path = os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4")
        checkpoint_path = './Wav2Lip/checkpoints/wav2lip_gan.pth'  # Update this path to your Wav2Lip checkpoint
        os.system(f'python ./Wav2Lip/inference.py --checkpoint_path {checkpoint_path} --face {video_temp_path} --audio {audio_temp_path} --outfile {lipsync_output_path} --nosmooth')

        res_motion.append(res_motion_current)
        np.savez(os.path.join(save_dir, f"audio_{idx}_retri_{counter}.npz"), motion=res_motion_current)
    
        start_node = path[1].index
        end_node = start_node + 100
    print(f"delete gt-nodes {start_node}, {end_node}")
    nodes_to_delete = list(range(start_node, end_node))
    graph.delete_vertices(nodes_to_delete)
    graph = graph_pruning(graph)
    path_list, is_continue_list = search_path_dp(graph, audio_low_all, audio_high_all, top_k=1, search_mode="both")
    res_motion = []
    counter = 1
    for path, is_continue in zip(path_list, is_continue_list):
        res_motion_current = path_visualization(
          graph, path, is_continue, os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4"), audio_path=audio_path, return_motion=True, verbose_continue=True
        )
        video_temp_path = os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4")
        
        video_reader = VideoReader(video_temp_path)
        video_np = []
        for i in range(len(video_reader)):
            if i == 0: continue
            video_frame = video_reader[i].asnumpy()
            video_np.append(Image.fromarray(video_frame))
        adjusted_video_pil = adjust_statistics_to_match_reference([video_np])
        save_videos_from_pil(adjusted_video_pil[0], os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4"), fps=30, bitrate=2000000)


        audio_temp_path = audio_path
        lipsync_output_path = os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4")
        checkpoint_path = './Wav2Lip/checkpoints/wav2lip_gan.pth'  # Update this path to your Wav2Lip checkpoint
        os.system(f'python ./Wav2Lip/inference.py --checkpoint_path {checkpoint_path} --face {video_temp_path} --audio {audio_temp_path} --outfile {lipsync_output_path} --nosmooth')
        res_motion.append(res_motion_current)
        np.savez(os.path.join(save_dir, f"audio_{idx}_retri_{counter}.npz"), motion=res_motion_current)
    
    result = [
        os.path.join(save_dir, f"audio_{idx}_retri_0.mp4"),
        os.path.join(save_dir, f"audio_{idx}_retri_1.mp4"),
        os.path.join(save_dir, f"audio_{idx}_retri_0.npz"),
        os.path.join(save_dir, f"audio_{idx}_retri_1.npz")
    ]
    return result


def init_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
       

def prepare_all(yaml_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=yaml_name)
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config.endswith(".yaml"):
        config = OmegaConf.load(args.config)
        config.exp_name = args.config.split("/")[-1][:-5]
    else:
        raise ValueError("Unsupported config file format. Only .yaml files are allowed.")
    save_dir = os.path.join(config.output_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    return config


def save_first_10_seconds(video_path, output_path="./save_video.mp4"):
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_to_save = fps * 10
    frame_count = 0
    
    while cap.isOpened() and frame_count < frames_to_save:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


character_name_to_yaml = {
  "speaker8_jjRWaMCWs44_00-00-30.16_00-00-33.32.mp4": "./datasets/data_json/youtube_test/speaker8.json",
  "speaker7_iuYlGRnC7J8_00-00-0.00_00-00-3.25.mp4": "./datasets/data_json/youtube_test/speaker7.json",
  "speaker9_o7Ik1OB4TaE_00-00-38.15_00-00-42.33.mp4": "./datasets/data_json/youtube_test/speaker9.json",
  "1wrQ6Msp7wM_00-00-39.69_00-00-45.68.mp4": "./datasets/data_json/youtube_test/speaker1.json",
  "101099-00_18_09-00_18_19.mp4": "./datasets/data_json/show_oliver_test/Stupid_Watergate_-_Last_Week_Tonight_with_John_Oliver_HBO-FVFdsl29s_Q.mkv.json",
}

@spaces.GPU(duration=240) 
def tango(audio_path, character_name, seed, create_graph=False, video_folder_path=None):
    cfg = prepare_all("./configs/gradio.yaml")
    cfg.seed = seed
    seed_everything(cfg.seed)
    experiment_ckpt_dir = experiment_log_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    saved_audio_path = "./saved_audio.wav"
    sample_rate, audio_waveform = audio_path 
    sf.write(saved_audio_path, audio_waveform, sample_rate)

    audio_waveform, sample_rate = librosa.load(saved_audio_path)
    # print(audio_waveform.shape)
    resampled_audio = librosa.resample(audio_waveform, orig_sr=sample_rate, target_sr=16000)
    required_length = int(16000 * (128 / 30)) * 2
    resampled_audio = resampled_audio[:required_length]
    sf.write(saved_audio_path, resampled_audio, 16000)
    audio_path = saved_audio_path
    
    yaml_name = character_name_to_yaml.get(character_name.split("/")[-1], "./datasets/data_json/youtube_test/speaker1.json")
    cfg.data.test_meta_paths = yaml_name
    print(yaml_name, character_name.split("/")[-1])

    if character_name.split("/")[-1] not in character_name_to_yaml.keys():
        create_graph=True
        # load video, and save it to "./save_video.mp4 for the first 20s of the video."
        os.makedirs("./outputs/tmpvideo/", exist_ok=True)
        save_first_10_seconds(character_name, "./outputs/tmpvideo/save_video.mp4")

    if create_graph:
        video_folder_path = "./outputs/tmpvideo/"
        data_save_path = "./outputs/tmpdata/"
        json_save_path = "./outputs/save_video.json"
        graph_save_path = "./outputs/save_video.pkl"
        os.system(f"cd ./SMPLer-X/ && python app.py --video_folder_path {video_folder_path} --data_save_path {data_save_path} --json_save_path {json_save_path} && cd ..")
        os.system(f"python ./create_graph.py --json_save_path {json_save_path} --graph_save_path {graph_save_path}") 
        cfg.data.test_meta_paths = json_save_path

    smplx_model = smplx.create(
        "./emage/smplx_models/", 
        model_type='smplx',
        gender='NEUTRAL_2020', 
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100, 
        ext='npz',
        use_pca=False,
    )
    model = init_class(cfg.model.name_pyfile, cfg.model.class_name, cfg)
    for param in model.parameters():
        param.requires_grad = False
    model.smplx_model = smplx_model
    model.get_motion_reps = get_motion_reps_tensor
    
    local_rank = 0  
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    smplx_model = smplx_model.to(device).eval()
    model = model.to(device)
    model.smplx_model = model.smplx_model.to(device)

    checkpoint_path = "./datasets/cached_ckpts/ckpt.pth"
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    test_path = os.path.join(experiment_ckpt_dir, f"test_{0}")
    os.makedirs(test_path, exist_ok=True)
    result = test_fn(model, device, 0, cfg.data.test_meta_paths, test_path, cfg, audio_path)
    gc.collect()
    torch.cuda.empty_cache()
    return result


examples_audio = [
    ["./datasets/cached_audio/example_male_voice_9_seconds.wav"],
    ["./datasets/cached_audio/example_female_voice_9_seconds.wav"],
]

examples_video = [
    ["./datasets/cached_audio/speaker8_jjRWaMCWs44_00-00-30.16_00-00-33.32.mp4"],
    ["./datasets/cached_audio/speaker7_iuYlGRnC7J8_00-00-0.00_00-00-3.25.mp4"],
    ["./datasets/cached_audio/speaker9_o7Ik1OB4TaE_00-00-38.15_00-00-42.33.mp4"],
    ["./datasets/cached_audio/1wrQ6Msp7wM_00-00-39.69_00-00-45.68.mp4"],
    ["./datasets/cached_audio/101099-00_18_09-00_18_19.mp4"],
]

combined_examples = [
    ["./datasets/cached_audio/example_male_voice_9_seconds.wav", "./datasets/cached_audio/speaker9_o7Ik1OB4TaE_00-00-38.15_00-00-42.33.mp4", 2024],
    ["./datasets/cached_audio/example_male_voice_9_seconds.wav", "./datasets/cached_audio/speaker7_iuYlGRnC7J8_00-00-0.00_00-00-3.25.mp4", 2024],
    ["./datasets/cached_audio/example_male_voice_9_seconds.wav", "./datasets/cached_audio/101099-00_18_09-00_18_19.mp4", 2024],
    ["./datasets/cached_audio/example_female_voice_9_seconds.wav", "./datasets/cached_audio/1wrQ6Msp7wM_00-00-39.69_00-00-45.68.mp4", 2024],
    ["./datasets/cached_audio/example_female_voice_9_seconds.wav", "./datasets/cached_audio/speaker8_jjRWaMCWs44_00-00-30.16_00-00-33.32.mp4", 2024],
]

def make_demo():
    with gr.Blocks(analytics_enabled=False) as Interface:
        # First row: Audio upload and Audio examples with adjusted ratio
        gr.Markdown(
            """
            <div align='center'> <h1> TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio Motion Embedding and Diffusion Interpolation </span> </h1> \
                        <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        <a href='https://h-liu1997.github.io/'>Haiyang Liu</a>, \
                        <a href='https://yangxingchao.github.io/'>Xingchao Yang</a>, \
                        <a href=''>Tomoya Akiyama</a>, \
                        <a href='https://sky24h.github.io/'> Yuantian Huang</a>, \
                        <a href=''>Qiaoge Li</a>, \
                        <a href='https://www.tut.ac.jp/english/university/faculty/cs/164.html'>Shigeru Kuriyama</a>, \
                        <a href='https://taketomitakafumi.sakura.ne.jp/web/en/'>Takafumi Taketomi</a>\
                    </h2> \
                    <a style='font-size:18px;color: #000000'>This is a preprint version, more details will be available at </a>\
                    <a style='font-size:18px;color: #000000' href=''>[Github Repo]</a>\
                        <a style='font-size:18px;color: #000000' href=''> [ArXiv] </a>\
                        <a style='font-size:18px;color: #000000' href='https://pantomatrix.github.io/TANGO/'> [Project Page] </a> </div>
                    </h2> \
                    <a style='font-size:18px;color: #000000'>This is an open-source project supported by Hugging Face's free ZeroGPU. Runtime is limited to 300s, so it operates in low-quality mode. Some high-quality mode results are shown below. </a> </div>
            """
        )

        # gr.Markdown("""
        # <h4 style="text-align: left;">
        # This demo is part of an open-source project supported by Hugging Face's free, zero-GPU runtime. Due to runtime cost considerations, it operates in low-quality mode. Some high-quality videos are shown below.

        # Details of the low-quality mode:
        # 1. Lower resolution.
        # 2. More discontinuous frames (causing noticeable "frame jumps").
        # 3. Utilizes open-source tools like SMPLerX-s-model, Wav2Lip, and FiLM for faster processing.
        # 4. Accepts audio input of up to 8 seconds. If your input exceeds 8 seconds, only the first 8 seconds will be used.
        # 5. You can provide a custom background video for your character, but it is limited to 20 seconds.

        # Feel free to open an issue on GitHub or contact the authors if this does not meet your needs.
        # </h4>
        # """)
        
        # Create a gallery with 5 videos
        with gr.Row():
            video1 = gr.Video(value="./datasets/cached_audio/demo1.mp4", label="Demo 0")
            video2 = gr.Video(value="./datasets/cached_audio/demo2.mp4", label="Demo 1")
            video3 = gr.Video(value="./datasets/cached_audio/demo3.mp4", label="Demo 2")
            video4 = gr.Video(value="./datasets/cached_audio/demo4.mp4", label="Demo 3")
            video5 = gr.Video(value="./datasets/cached_audio/demo5.mp4", label="Demo 4")
        with gr.Row():
            video1 = gr.Video(value="./datasets/cached_audio/demo6.mp4", label="Demo 5")
            video2 = gr.Video(value="./datasets/cached_audio/demo0.mp4", label="Demo 6")
            video3 = gr.Video(value="./datasets/cached_audio/demo7.mp4", label="Demo 7")
            video4 = gr.Video(value="./datasets/cached_audio/demo8.mp4", label="Demo 8")
            video5 = gr.Video(value="./datasets/cached_audio/demo9.mp4", label="Demo 9")


        with gr.Row():
            with gr.Column(scale=4):
                video_output_1 = gr.Video(label="Generated video - 1",
                            interactive=False,
                            autoplay=False,
                            loop=False,
                            show_share_button=True)
            with gr.Column(scale=4):
                video_output_2 = gr.Video(label="Generated video - 2",
                            interactive=False,
                            autoplay=False,
                            loop=False,
                            show_share_button=True)
            with gr.Column(scale=1):
                file_output_1 = gr.File(label="Download 3D Motion and Visualize in Blender")
                file_output_2 = gr.File(label="Download 3D Motion and Visualize in Blender")
                gr.Markdown("""
                <h4 style="text-align: left;">
                <a style='font-size:18px;color: #000000'> Details of the low-quality mode: </a>
                <br>
                <a style='font-size:18px;color: #000000'> 1. Lower resolution.</a>
                <br>
                <a style='font-size:18px;color: #000000'> 2. More discontinuous graph nodes (causing noticeable "frame jumps"). </a>
                <br>
                <a style='font-size:18px;color: #000000'> 3. Utilizes open-source tools like SMPLerX-s-model, Wav2Lip, and FiLM for faster processing. </a>
                <br>
                <a style='font-size:18px;color: #000000'> 4. only use first 8 seconds of your input audio.</a>
                <br>
                <a style='font-size:18px;color: #000000'> 5. custom character for a video up to 10 seconds. </a>
                <br>
                <br>
                <a style='font-size:18px;color: #000000'> Feel free to open an issue on GitHub or contact the authors if this does not meet your needs.</a>
                </h4>
                """)
            
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="Upload your audio")
                seed_input = gr.Number(label="Seed", value=2024, interactive=True)
            with gr.Column(scale=2):
                gr.Examples(
                    examples=examples_audio,
                    inputs=[audio_input],
                    outputs=[video_output_1, video_output_2, file_output_1, file_output_2],
                    label="Select existing Audio examples",
                    cache_examples=False
                )
            with gr.Column(scale=1):
                video_input = gr.Video(label="Your Character", elem_classes="video")
            with gr.Column(scale=2):
                gr.Examples(
                    examples=examples_video,
                    inputs=[video_input],  # Correctly refer to video input
                    outputs=[video_output_1, video_output_2, file_output_1, file_output_2],
                    label="Character Examples",
                    cache_examples=False
                )
        
        # Fourth row: Generate video button
        with gr.Row():
            run_button = gr.Button("Generate Video")
        
        # Define button click behavior
        run_button.click(
            fn=tango,
            inputs=[audio_input, video_input, seed_input],
            outputs=[video_output_1, video_output_2, file_output_1, file_output_2]
        )

        with gr.Row():
            with gr.Column(scale=4):
                print(combined_examples)
                gr.Examples(
                    examples=combined_examples,
                    inputs=[audio_input, video_input, seed_input],  # Both audio and video as inputs
                    outputs=[video_output_1, video_output_2, file_output_1, file_output_2],
                    fn=tango,  # Function that processes both audio and video inputs
                    label="Select Combined Audio and Video Examples (Cached)",
                    cache_examples=True
                )

    return Interface
      
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='8675'
    # #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    demo = make_demo()
    demo.launch(share=True)