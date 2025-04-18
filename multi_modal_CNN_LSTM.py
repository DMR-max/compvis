import os
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
import lmdb
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from ultralytics import YOLO
from predict_images import ImageRegressor, main, inference
from PIL import Image



from tqdm import tqdm
# Global variable to be set in main for worker initialization
SESSION_IDS = None
os.environ["TEMP"] = "G:/computer vision opdracht/temp"


def init_vid_extr(session_id, vid_map="RGB_videos/video_data_n3", H_map="RGB_videos/video_data_n3"):
    pth = os.path.join(vid_map, f"{session_id}.mp4")
    h_path = os.path.join(H_map, f"H{session_id}.mat")
    cap = cv2.VideoCapture(pth)
    # load homograph
    H = loadmat(h_path)['H']
    return cap, H

def world_to_pixel(wrld_cords, H):
    pts = wrld_cords.reshape(-1, 1, 2)
    pixel_pts = cv2.perspectiveTransform(pts, H)
    return pixel_pts.reshape(-1, 2)
    
def worker_init_fn(worker_id):
    # init video extr for process
    global vid_extr
    vid_extr = {sid: init_vid_extr(sid) for sid in range(8)}


def get_video_patch(frame_no, pos, session_id, video_size=64):
    global vid_extr
    if 'vid_extr' not in globals():
        vid_extr = {sid: init_vid_extr(sid) for sid in range(8)}
    # Get the video capture ignore H since pos is already in pixel coordinates.
    cap, _ = vid_extr[session_id]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        patch = np.zeros((video_size, video_size, 3), dtype=np.uint8)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use the provided pixel coordinate directly.
        x_img, y_img = pos
        patch = cv2.getRectSubPix(frame, (video_size, video_size), (x_img, y_img))
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return patch

def get_video_patch_yolo(frame_no, pos, session_id, video_size=64, yolo_model=None):
    global vid_extr
    if 'vid_extr' not in globals():
        vid_extr = {sid: init_vid_extr(sid) for sid in range(8)}
    
    cap, _ = vid_extr[session_id]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    
    if not ret:
        patch = np.zeros((video_size, video_size, 3), dtype=np.uint8)
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        return patch
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO model to detect persons
    results = yolo_model(frame_rgb, classes=[0], verbose=False)
    
    
    person_boxes = [box.xyxy.cpu().numpy() for box in results[0].boxes]
    
    if not person_boxes:  # No person detected
        patch = np.zeros((video_size, video_size, 3), dtype=np.uint8)
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        return patch
    
    person_boxes = np.vstack(person_boxes)

    
    person_bottom_centers = [
        ((b[0] + b[2]) / 2, b[3], b)  # (center_x, bottom_y, bounding_box)
        for b in person_boxes
    ]
    
    # Select the box closest to the given position and assume its the correct person.
    closest_box_data = min(person_bottom_centers, key=lambda c: (c[0] - pos[0]) ** 2 + (c[1] - pos[1]) ** 2)
    closest_box = closest_box_data[2]

    x_min, y_min, x_max, y_max = map(int, closest_box)
    bottom_center = np.float32([closest_box_data[0], closest_box_data[1], 0])
    # Crop the detected person
    patch = frame_rgb[y_min:y_max, x_min:x_max]
    
    # Resize to the required video_size
    patch = cv2.resize(patch, (video_size, video_size))
    
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1)) 

    
    return patch, bottom_center


class TrajectoryVideoDataset(Dataset):
    def __init__(self, label_file, session_id, timesteps=10, video_size=64, door_y=0, use_precomputed=False, hdf5_path=None, yolo_model=None):
        self.yolo_model = yolo_model
        self.sess_id = session_id
        self.t = timesteps
        self.vid_size = video_size
        self.door_y = door_y
        self.data = pd.read_csv(label_file, header=None, names=['frame', 'person_id', 'x', 'y', 'z'])
        self.data = self.data.sort_values(by=['person_id', 'frame']).reset_index(drop=True)
        self.samples = []
        for pid, group in self.data.groupby('person_id'):
            group = group.sort_values('frame').reset_index(drop=True)
            if len(group) < timesteps + 1:
                continue
            for i in range(len(group) - timesteps):
                seq = group.iloc[i:i+timesteps]
                target = group.iloc[i+timesteps]
                self.samples.append((seq, target))
        self.num_samples = len(self.samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq_samples, target_samples = self.samples[idx]
        # Get the homography matrix from the video extr
        cap, H = init_vid_extr(self.sess_id)
        
        # Convert target positions from world to pixel coordinates
        target_samples_pos = target_samples[['x','y','z']].values.astype(np.float32)

        # Convert the input sequence positions
        pos_seq_samples = seq_samples[['x','y','z']].values.astype(np.float32)
        pixel_coords = world_to_pixel(pos_seq_samples[:, :2], H)
        pos_seq_samples_vid = pos_seq_samples.copy()
        pos_seq_samples_vid[:, :2] = pixel_coords

        if self.door_y is not None:
            door_feature = (seq_samples[['y']].values.astype(np.float32) - self.door_y)
            pos_seq_samples = np.concatenate([pos_seq_samples, door_feature], axis=1)

        vid_seq = []
        # Use the pixel_coords array (already converted) when extracting patches
        for i, (_, row) in enumerate(seq_samples.iterrows()):
            frame_no = int(row['frame'])
            pos_xy = pixel_coords[i]  
            patch = get_video_patch_yolo(frame_no, pos_xy, self.sess_id, video_size=self.vid_size, yolo_model=self.yolo_model)[0]
            vid_seq.append(patch)
        vid_seq = np.stack(vid_seq, axis=0)  # (T, channels, video_size, video_size)
    
        return pos_seq_samples, vid_seq, target_samples_pos

class TrajVideoCNNLSTM(nn.Module):
    def __init__(self, pos_dim=6, pos_embed_dim=16, video_channels=3, video_size=128,
                 cnn_feature_dim=32, lstm_hidden_dim=64, lstm_layers=1, output_dim=3):
        super().__init__()
        # Embed the trajectory positions
        self.pos_fc = nn.Linear(pos_dim, pos_embed_dim)
        
        # CNN encoder for video patches
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(video_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (16, video_size/2, video_size/2)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # Output: (32, video_size/4, video_size/4)
        )
        cnn_out_size = 32 * (video_size // 4) * (video_size // 4)
        self.video_fc = nn.Linear(cnn_out_size, cnn_feature_dim)
        
        # Comb the pos and vid features and project them before the LSTM
        self.input_fc = nn.Linear(pos_embed_dim + cnn_feature_dim, lstm_hidden_dim)
        
        # LSTM to process the sequential data
        self.lstm = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        
        # Final layer to pred output trajector
        self.fc_out = nn.Linear(lstm_hidden_dim, output_dim)
        
    def forward(self, pos_seq, vid_seq, dpth_seq=None, dir_seq=None,):
        batch, T, _ = pos_seq.shape
        
        # Process traj pos
        pos_emb = self.pos_fc(torch.cat([pos_seq, dpth_seq, dir_seq], dim=2))  # shape: (batch, T, pos_embed_dim)
        
        # Reshape to combine batch and T dimensions for CNN processing.
        vid_seq = vid_seq.view(batch * T, vid_seq.shape[2], vid_seq.shape[3], vid_seq.shape[4])
        cnn_out = self.cnn_encoder(vid_seq)  # shape: (batch*T, 32, video_size/4, video_size/4)
        cnn_out = cnn_out.contiguous().view(batch, T, -1)  # shape: (batch, T, cnn_out_size)
        vid_emb = self.video_fc(cnn_out)  # shape: (batch, T, cnn_feature_dim)
        
        # Concat pos and vid features along the feature dimension
        combined = torch.cat([pos_emb, vid_emb], dim=2)  # shape: (batch, T, pos_embed_dim+cnn_feature_dim)
        
        # Project the concat features to the LSTM input dimension
        lstm_input = self.input_fc(combined)  # shape: (batch, T, lstm_hidden_dim)
        
        lstm_out, (h_n, _) = self.lstm(lstm_input)  # h_n: (num_layers, batch, lstm_hidden_dim)
        
        # Use the last layer h state (for each sample in batch)
        last_hidden = h_n[-1]  # shape: (batch, lstm_hidden_dim)
        
        # final pred
        pred = self.fc_out(last_hidden)  # shape: (batch, output_dim)
        return pred


class TrajVideoTransformer(nn.Module):
    def __init__(self, pos_dim=6, pos_embed_dim=16, video_channels=3, video_size=128, cnn_feature_dim=32, transformer_dim=64, nhead=8, num_layers=3, output_dim=3):
        super().__init__()
        # embed trajectory
        self.pos_fc = nn.Linear(pos_dim, pos_embed_dim)
        # CNN encoder for video patch
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(video_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, video_size/2, video_size/2)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # (32, video_size/4, video_size/4)
        )
        cnn_out_size = 32 * (video_size // 4) * (video_size // 4)
        self.video_fc = nn.Linear(cnn_out_size, cnn_feature_dim)
        # combine both features time step add project to transformer dimension
        self.input_fc = nn.Linear(pos_embed_dim + cnn_feature_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # final pred
        self.fc_out = nn.Linear(transformer_dim, output_dim)
        
    def forward(self, pos_seq, vid_seq, dpth_seq=None, dir_seq=None,):
        batch, T, _ = pos_seq.shape
        pos_emb = self.pos_fc(torch.cat([pos_seq, dpth_seq, dir_seq], dim=2))  # (batch, T, pos_embed_dim) (shape commented for clarity)
        # proc video patch
        vid_seq = vid_seq.view(batch * T, vid_seq.shape[2], vid_seq.shape[3], vid_seq.shape[4])
        cnn_out = self.cnn_encoder(vid_seq)  # (batch*T, 32, video_size/4, video_size/4)
        cnn_out = cnn_out.contiguous().view(batch, T, -1)
        vid_emb = self.video_fc(cnn_out)  # (batch, T, cnn_feature_dim)
        # Concat feature dimension
        combined = torch.cat([pos_emb, vid_emb], dim=2)  # (batch, T, pos_embed_dim+cnn_feature_dim)
        x = self.input_fc(combined)  # (batch, T, transformer_dim)
        # Transformer expect shape (T, batch, transformer_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        # Use the output at the last time step for prediction
        out = x[-1, :, :]  # (batch, transformer_dim)
        pred = self.fc_out(out)  # (batch, output_dim)
        return pred
    
def precompute_and_save_video_patches_lmdb(dataset, lmdb_output_path, batch_size=64, map_size=int(38 * 1024**3), sess_id=None):
    if sess_id == 5:
        map_size = int(150 * 1024**3)
    if sess_id == 3:
        map_size = int(200 * 1024**3)
    if sess_id == 4:
        map_size = int(250 * 1024**3)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    total_samples = len(dataset)
    env = lmdb.open(lmdb_output_path, map_size=map_size)
    # Store the total number of samples.
    with env.begin(write=True) as txn:
        txn.put(b'num_samples', pickle.dumps(total_samples))
    sample_index = 0
    for pos_b, vid_b, target_b in tqdm(loader, desc="Precomputing LMDB patches"):
        b_size = pos_b.size(0)
        with env.begin(write=True) as txn:
            for i in range(b_size):
                # Convert the video patch from float32 [0,1] to uint8 [0,255]
                vid_patch = vid_b[i].numpy() *255
                vid_patch_uint8 = (vid_patch).astype(np.uint8)
                sample = {
                    'pos_seq': pos_b[i].numpy(),
                    'vid_seq': vid_patch_uint8,           
                    'target': target_b[i].numpy()       
                }
                key = f'sample_{sample_index:08d}'.encode('ascii')
                txn.put(key, pickle.dumps(sample))
                sample_index += 1
    env.close()

class TrajectoryVideoLMDBDataset(Dataset):
    def __init__(self, lmdb_path, sess_id, door_y=None, norm=False, f_width=None, f_height=None):
        self.norm = norm
        self.f_width = f_width
        self.f_height = f_height
        self.door_y = door_y
        self.lmdb_path = lmdb_path
        self.sess_id = sess_id
        self.env = None
        # Open the LMDB environment in read-only mode to fetch the total number of samples
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.num_samples = pickle.loads(txn.get(b'num_samples'))
        # Retrieve the homography matrix for this session
        _, self.H = init_vid_extr(self.sess_id)

        # If norm is req determine frame dim
        if self.norm:
            if f_width is None or f_height is None:
                cap, _ = init_vid_extr(self.sess_id)
                self.f_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.f_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            else:
                self.f_width = f_width
                self.f_height = f_height
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            key = f'sample_{idx:08d}'.encode('ascii')
            sample = pickle.loads(txn.get(key))
        # Load the position sequences and target in world coordinates
        pos_seq = torch.tensor(sample['pos_seq'], dtype=torch.float32)
        target_pos = torch.tensor(sample['target'], dtype=torch.float32)
        # Convert video patch back to float32 [0,1]
        vid_seq = torch.tensor(sample['vid_seq'], dtype=torch.float32) / 255.0

        # Convert the first two cols of pos_seq and target from world to pixel coordinates.
        pos_seq_np = pos_seq.numpy()
        target_pos_np = target_pos.numpy()
        # For pos_seq, assume it's 2D (T, 3)
        pos_seq_np[:, :2] = world_to_pixel(pos_seq_np[:, :2], self.H)
        pos_seq_np[:, 3] = pos_seq_np[:, 1] - self.door_y
        # Reshape to (1,2) for conversion then assign back.
        converted_to_pixel_pos = world_to_pixel(target_pos_np[:2].reshape(1, 2), self.H)  # shape (1,2)
        target_pos_np[:2] = converted_to_pixel_pos[0]

        # if norm requested, scale the x and y coordinates by the frame dimensions.
        if self.norm:
            pos_seq_np[:, 0] /= self.f_width
            pos_seq_np[:, 1] /= self.f_height
            pos_seq_np[:, 3] /= self.f_height
            target_pos_np[:2] /= np.array([self.f_width, self.f_height])
        # conv back to torch tensors.
        pos_seq = torch.tensor(pos_seq_np, dtype=torch.float32)
        target = torch.tensor(target_pos_np, dtype=torch.float32)
        
        return pos_seq, vid_seq, target

    
def build_holdout_dict(label_file, session_id, timesteps=10, holdout_steps=10, video_size=128, door_y=None, yolo_model=None):
    # Initialize the video extractor to load H
    cap, H = init_vid_extr(session_id)
    
    df = pd.read_csv(label_file, header=None, names=['frame', 'person_id', 'x', 'y', 'z'])
    df_sorted = df.sort_values(by=['person_id', 'frame']).reset_index(drop=True)
    holdout_list = {}
    
    for id, grp in tqdm(df_sorted.groupby('person_id')):
        grp = grp.sort_values('frame').reset_index(drop=True)
        if len(grp) < timesteps + holdout_steps:
            continue
        # load positions from csv in world coordinates
        pos = grp[['x', 'y', 'z']].values.astype(np.float32)

        # convert the world coordinates to pixel coordinates using H
        pos_pixel = pos.copy()
        pos_pixel[:, :2] = world_to_pixel(pos[:, :2], H)
        
        frames = grp['frame'].values
        # se the transformed (pixel) positions for the initial sequence
        init_seq_pos = []
        init_seq_pos = pos_pixel[-(timesteps+holdout_steps):-holdout_steps]
        door_feature = (init_seq_pos[:, 1:2] - door_y)
        init_seq_pos = np.concatenate([init_seq_pos, door_feature], axis=1)
        
        init_seq_vid = []
        # extract video patches for each frame in the input sequence.
        # here we use the pixel positions directly.
        for i in range(len(grp) - (timesteps + holdout_steps), len(grp) - holdout_steps):
            frame_no = int(frames[i])
            # Use the transformed pixel (x, y)
            pos_xy = pos_pixel[i, :2]
            patch, _ = get_video_patch_yolo(frame_no, pos_xy, session_id, video_size, yolo_model=yolo_model)

            init_seq_vid.append(patch)
            
        # init_seq_pos = np.stack(init_seq_pos, axis=0)
        init_seq_vid = np.stack(init_seq_vid, axis=0)  # shape: (timesteps, channels, video_size, video_size)
        # Transform the true future positions as well.
        true_future = pos_pixel[-holdout_steps:]
        holdout_list[id] = (init_seq_pos, init_seq_vid, true_future)
    
    return holdout_list

def predict_future_steps(model, init_seq_pos, init_seq_vid, num_steps, door_y=None, depth_map=None, angle_model=None):
    model.eval()
    pred_list = []
    curr_pos = torch.tensor(init_seq_pos).unsqueeze(0).to(device)  # (1, T, pos_dim)
    curr_vid = torch.tensor(init_seq_vid).unsqueeze(0).to(device)  # (1, T, C, H, W) 
     

    
    

    for i in tqdm(range(num_steps)):
        
        curr_angle = inference(angle_model, curr_vid, batch_size=1, frame_size=10)
        x_values= curr_pos[..., 1]
        y_values = curr_pos[..., 0]
            
        x_indices = x_values.type(torch.int64)
        y_indices = y_values.type(torch.int64)
        
        curr_dpth = depth_map[x_indices, y_indices].unsqueeze(-1)
        
        with torch.no_grad():
            pred = model(curr_pos, curr_vid, curr_dpth, curr_angle)
        
        pred_door = pred[:, 1:2] - door_y  # shape: (1,1)
        pred_full = torch.cat([pred, pred_door], dim=1)  # shape: (1,4)
        
        pred_np = pred_full.squeeze(0).cpu().numpy()  # shape: (output_dim) or (output_dim+1)
        pred_list.append(pred_np)
        
        # Update current_pos remove the oldest time step and append the new prediction.
        curr_pos = torch.cat([curr_pos[:, 1:, :], pred_full.unsqueeze(1)], dim=1)
        
        # For the video sequence, we reuse the last patch.
        last_patch = curr_vid[:, -1:, :, :, :]
        curr_vid = torch.cat([curr_vid[:, 1:, :, :, :], last_patch], dim=1)
        
    
    return np.array(pred_list)


def calculate_metrics(predicted, true):
    predicted = predicted[:, :true.shape[-1]]
    
    mse = np.mean((predicted - true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted - true))
    return mse, rmse, mae

def inverse_normalize_coords(coords, frame_width, frame_height):
    # Assuming coords is a 2D array with at least 2 columns
    coords[:, 0] *= frame_width
    coords[:, 1] *= frame_height
    coords[:, 3] *= frame_height
    return coords

def evaluate_holdout(model, holdout_dict, holdout_steps=10, selected_persons=None, door_y=None, frame_width=None, frame_height=None, depth_map=None, angle_model=None):
    model.eval()
    results = {}
    metrics_all = []

    for pid, (init_seq_pos, init_seq_vid, true_future) in holdout_dict.items():
        if selected_persons is not None and pid not in selected_persons:
            continue

        # Predict future steps
        pred_future = predict_future_steps(model, init_seq_pos, init_seq_vid, holdout_steps, door_y=door_y, depth_map=depth_map, angle_model=angle_model)

        if frame_width is not None and frame_height is not None:
            pred_future_pixel = inverse_normalize_coords(pred_future.copy(), frame_width, frame_height)
            true_future_pixel = inverse_normalize_coords(true_future.copy(), frame_width, frame_height)
        else:
            pred_future_pixel = pred_future
            true_future_pixel = true_future
        
        # Calculate metrics
        mse, rmse, mae = calculate_metrics(pred_future_pixel, true_future_pixel)
        metrics_all.append((mse, rmse, mae))
        
        # Print results for this person.
        print(f"person {pid}:")
        print("predicted Future Pos:")
        print(pred_future_pixel)
        print("true Future Pos:")
        print(true_future_pixel)
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        # scheidings lijn
        print("-" * 50)
        
        results[pid] = (pred_future_pixel, true_future_pixel)
    
    # Aggregate metrics over all persons.
    if metrics_all:
        metrics_all = np.array(metrics_all)
        avg_mse = np.mean(metrics_all[:, 0])
        avg_rmse = np.mean(metrics_all[:, 1])
        avg_mae = np.mean(metrics_all[:, 2])
        print("Overall Metrics on Holdout Data:")
        print(f"Avg MSE: {avg_mse:.4f}, Avg RMSE: {avg_rmse:.4f}, Avg MAE: {avg_mae:.4f}")
    
    return results

def loadZoe():
    # Load latest version of midas first
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    repo = "isl-org/ZoeDepth"
    
    return torch.hub.load(repo, "ZoeD_NK", pretrained=True)


###############################################################################
# 5. MAIN
###############################################################################
if __name__ == '__main__':
    DOOR_Y = 450

    # List of sessions and corresponding label files
    session_ids = [1, 2, 3, 4, 5, 6, 7] 
    label_files = [f"Pedestrian_labels/{sid}_frame.txt" for sid in session_ids]
    
    # Set global SESSION_IDS for worker initialization
    SESSION_IDS = session_ids

    timesteps = 10
    v_size = 128
    b_size = 128
    num_epochs = 50
    
    lmdb_folder = "G:/computer vision opdracht/PrecomputedLMDB"
    # lmdb_folder = "C:/cvisdata/PrecomputedLMDB"

    # LMDB file names for each session (one file per session)
    db_files = {sid: os.path.join(lmdb_folder, f"precomputed_patches_session_{sid}.lmdb") for sid in session_ids}
    os.makedirs(lmdb_folder, exist_ok=True)

    d_list = []
    yolo_model = YOLO("yolo11n.pt")
    # For each session check if the LMDB exists
    for label_file, sid in zip(label_files, session_ids):
        lmdb_path = db_files[sid]
        if not os.path.exists(lmdb_path):
            print(f"Precomputing video patches for session {sid} and saving to {lmdb_path}...")
            temp_dataset = TrajectoryVideoDataset(label_file, sid, timesteps=timesteps, video_size=v_size, door_y=DOOR_Y, yolo_model=yolo_model)
            precompute_and_save_video_patches_lmdb(temp_dataset, lmdb_path, batch_size=128, sess_id=sid)
        # Create LMDB dataset.
        ds = TrajectoryVideoLMDBDataset(lmdb_path, sid, door_y=DOOR_Y, norm=False)
        d_list.append(ds)
    t_dataset = ConcatDataset(d_list)
    print(f"Number of training samples: {len(t_dataset)}")
    print("Building training dataloader...")
    t_loader = DataLoader(t_dataset, batch_size=b_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = TrajVideoTransformer().to(device)
    # model = TrajVideoCNNLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # checkpoint_path = "checkpoint_CNN_LAST_good_norm_epoch_50.pth"
    checkpoint_path = "checkpoint_trans_good_ALL_POINTS_epoch_22.pth"
    start_epoch = 0 

    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"resuming training from epoch {start_epoch}")
        epoch = start_epoch

    #Training loop
    model.train()
    print("Starting training...")
    
    #Get specific frame for zoe
    global vid_extr
    if 'vid_extr' not in globals():
        vid_extr = {sid: init_vid_extr(sid) for sid in range(8)}
    cap, H = vid_extr[3]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 7200) #120*60 should be at 2:00 session 3, empty hallway.
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #(y, x c)

    #Run zoe once
    zoe = loadZoe()
    depth_map = zoe.infer_pil(frame, output_type="tensor").to(device)
    angle_model = ImageRegressor().to(device)
    angle_checkpoint = torch.load(os.path.join(os.getcwd(), f"modified_angle\checkpoint_angle_pred_images_TBD_epoch_{20}.pth"), map_location=device)
    angle_model.load_state_dict(angle_checkpoint)

    for epoch in range(num_epochs):
        real_epoch = start_epoch + epoch
        checkpoint_path = f"checkpoint_trans_good_ALL_POINTS_epoch_{real_epoch+1}.pth"
        epoch_loss = 0.0
        for pos_seq, vid_seq, target in tqdm(t_loader):
            angle_results = inference(angle_model, vid_seq)

            
            pos_seq = pos_seq.to(device) 
            vid_seq = vid_seq.to(device)
            
            
            orig_width, orig_height = 1280, 1024
            depth_H, depth_W = depth_map.shape[-2:] 

            scale_x = depth_W / orig_width 
            scale_y = depth_H / orig_height 

            y_indices = pos_seq[..., 0].round().float() * scale_y
            x_indices = pos_seq[..., 1].round().float() * scale_x

            y_indices = y_indices.round().to(torch.int64).clamp(0, depth_H - 1)
            x_indices = x_indices.round().to(torch.int64).clamp(0, depth_W - 1)

            if depth_map.ndim == 3 and depth_map.shape[0] == 1:
                depth_map = depth_map.squeeze(0)

            dpth_seq = depth_map[y_indices, x_indices].unsqueeze(-1)

            target = target.to(device)
            optimizer.zero_grad()
            output = model(pos_seq, vid_seq, dpth_seq, angle_results).to(device)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * pos_seq.size(0)
        epoch_loss_avg = epoch_loss / len(t_dataset)
        print(f"Epoch {real_epoch+1}/{num_epochs} Loss: {epoch_loss/len(t_dataset):.4f}")
            # Save checkpoint after each epoch
        torch.save({
            'epoch': real_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss_avg,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {real_epoch+1}")
    
    # Build holdout dictionary from the same label file (or a different one for evaluation)
    #global yolo_model 
    holdout_steps = 10
    holdout_dict = build_holdout_dict("Pedestrian_labels/0_frame.txt", 0, timesteps=timesteps, holdout_steps=holdout_steps, video_size=v_size, door_y=DOOR_Y, yolo_model=yolo_model)
    
    # Evaluate on selected persons (set selected_persons to None to evaluate all)
    selected_persons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    eval_results = evaluate_holdout(model, holdout_dict, holdout_steps=holdout_steps, selected_persons=None, door_y=DOOR_Y, depth_map=depth_map, angle_model=angle_model)