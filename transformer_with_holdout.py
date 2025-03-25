import os
import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

def init_vid_extr(session_id, vid_map="RGB_videos/video_data_n3", H_map="RGB_videos/video_data_n3"):
    pth = os.path.join(vid_map, f"{session_id}.mp4")
    h_path = os.path.join(H_map, f"H{session_id}.mat")
    cap = cv2.VideoCapture(pth)
    # load homograph
    H = loadmat(h_path)['H']
    return cap, H

def worker_init_fn(worker_id):
    # init video extr for process
    global video_extr
    video_extr = {sid: init_vid_extr(sid) for sid in range(8)}


def get_video_patch(frame_no, pos, session_id, video_size=64):
    global video_extr
    if 'video_extractors' not in globals():
        vid_extr = {sid: init_vid_extr(sid) for sid in range(8)}
    cap, H = vid_extr[session_id]
    # jump to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    # Read frame
    returns_error, frame = cap.read()
    # check if frame read correctly
    if not returns_error:
        patch = np.zeros((video_size, video_size, 3), dtype=np.uint8)
    else:
        # convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Prepare the world point (only use x,y) depth is not needed
        point = np.array([[pos]], dtype=np.float32)  # shape (1,1,2)
        point_img = cv2.perspectiveTransform(point, H)  # transforms to image coordinates
        x_img, y_img = point_img[0, 0]
        # Extract patch centered at (x_img, y_img)
        patch = cv2.getRectSubPix(frame, (video_size, video_size), (x_img, y_img))
    patch = patch.astype(np.float32) / 255.0
    # Change shape to (channels, video_size, video_size)
    patch = np.transpose(patch, (2, 0, 1))
    return patch

# from functools import lru_cache

# @lru_cache(maxsize=100)  # Adjust maxsize based on available memory
# def load_frame(frame_no, session_id, video_size):
#     cap, H = init_video_extractor(session_id)
#     # Make sure frame_no is a float for OpenCV
#     cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_no))
#     ret, frame = cap.read()
#     cap.release()
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     else:
#         # Return a blank frame if reading fails
#         frame = np.zeros((video_size, video_size, 3), dtype=np.uint8)
#     return frame, H

# def extract_patch_from_frame(frame, pos, H, video_size=64):
#     point = np.array([[pos]], dtype=np.float32)  # shape (1,1,2)
#     point_img = cv2.perspectiveTransform(point, H)  # map to image coordinates
#     x_img, y_img = point_img[0, 0]
#     patch = cv2.getRectSubPix(frame, (video_size, video_size), (x_img, y_img))
#     patch = patch.astype(np.float32) / 255.0
#     patch = np.transpose(patch, (2, 0, 1))
#     return patch

class TrajectoryVideoDataset(Dataset):
    def __init__(self, label_file, session_id, timesteps=10, video_size=64):
        self.sess_id = session_id
        self.t = timesteps
        self.vid_size = video_size
        self.data = pd.read_csv(label_file, header=None, names=['frame', 'person_id', 'x', 'y', 'z'])
        # sort by person and frame
        self.data = self.data.sort_values(by=['person_id', 'frame']).reset_index(drop=True)
        self.samples = []
        self.sess_id = session_id
        # build sequences per person
        for _, group in self.data.groupby('person_id'):
            group = group.sort_values('frame').reset_index(drop=True)
            if len(group) < timesteps + 1:
                continue
            for i in range(len(group) - timesteps):
                seq = group.iloc[i:i+timesteps]
                target = group.iloc[i+timesteps]
                self.samples.append((seq, target))
        self.frame_cache = {}  
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_samples, target_samples = self.samples[idx]
        # trajectory positions: use (x,y,z)
        pos_seq_samples = seq_samples[['x','y','z']].values.astype(np.float32)  # shape (timesteps, 3)
        target_samples_pos = target_samples[['x','y','z']].values.astype(np.float32)  # shape (3,)
        # for vid extract a patch  using the (x,y) coordinat
        vid_seq = []
        for _, row in seq_samples.iterrows():
            frame_no = int(row['frame'])
            pos_xy = np.array([row['x'], row['y']])
            patch = get_video_patch(frame_no, pos_xy, self.sess_id, video_size=self.vid_size)
            vid_seq.append(patch)
        vid_seq = np.stack(vid_seq, axis=0)  # shape (timesteps, channels, video_size, video_size)
        return torch.tensor(pos_seq_samples), torch.tensor(vid_seq), torch.tensor(target_samples_pos)

class TrajVideoTransformer(nn.Module):
    def __init__(self, pos_dim=3, pos_embed_dim=16, video_channels=3, video_size=64, cnn_feature_dim=32, transformer_dim=64, nhead=8, num_layers=3, output_dim=3):
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
        
    def forward(self, pos_seq, vid_seq):
        # pos_seq shape (batch, T, pos_dim)
        # vid_seq shape (batch, T, channels, video_size, video_size)
        batch, T, _ = pos_seq.shape
        # proc pos
        pos_emb = self.pos_fc(pos_seq)  # (batch, T, pos_embed_dim)
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

def build_holdout_dict(label_file, session_id, timesteps=10, holdout_steps=10, video_size=64):
    df = pd.read_csv(label_file, header=None, names=['frame', 'person_id', 'x', 'y', 'z'])
    df = df.sort_values(by=['person_id', 'frame']).reset_index(drop=True)
    holdout = {}
    for pid, group in tqdm(df.groupby('person_id')):
        group = group.sort_values('frame').reset_index(drop=True)
        if len(group) < timesteps + holdout_steps:
            continue
        positions = group[['x','y','z']].values.astype(np.float32)
        frames = group['frame'].values.astype(np.int32)
        init_seq_pos = positions[-(timesteps+holdout_steps):-holdout_steps]
        init_seq_vid = []
        # extract video patch for each frame in the input sequence
        for i in range(len(group) - (timesteps + holdout_steps), len(group) - holdout_steps):
            frame_no = int(frames[i])
            pos_xy = np.array([group.iloc[i]['x'], group.iloc[i]['y']])
            patch = get_video_patch(frame_no, pos_xy, session_id, video_size)
            init_seq_vid.append(patch)
        init_seq_vid = np.stack(init_seq_vid, axis=0)  # shape: (timesteps, channels, video_size, video_size)
        true_future = positions[-holdout_steps:]
        holdout[pid] = (init_seq_pos, init_seq_vid, true_future)
    return holdout

def predict_future_steps(model, init_seq_pos, init_seq_vid, num_steps):
    model.eval()
    predictions = []
    # convert initial sequences to tensors and add batch dimension
    current_pos = torch.tensor(init_seq_pos, dtype=torch.float32).unsqueeze(0).to(device)
    current_vid = torch.tensor(init_seq_vid, dtype=torch.float32).unsqueeze(0).to(device)
    for i in tqdm(range(num_steps)):
        with torch.no_grad():
            pred = model(current_pos, current_vid)  # shape: (1, output_dim)
        pred_np = pred.squeeze(0).cpu().numpy()  # shape: (output_dim,)
        predictions.append(pred_np)
        # update pos seq drop the oldest and append the prediction
        current_pos = torch.cat([current_pos[:, 1:, :], pred.unsqueeze(1)], dim=1)
        # for video reuse the last patch.
        last_patch = current_vid[:, -1:, :, :, :]
        current_vid = torch.cat([current_vid[:, 1:, :, :, :], last_patch], dim=1)
    return np.array(predictions)

def evaluate_holdout(model, holdout_dict, holdout_steps=10, selected_persons=None):
    model.eval()
    results = {}
    for pid, (init_seq_pos, init_seq_vid, true_future) in holdout_dict.items():
        if selected_persons is not None and pid not in selected_persons:
            continue
        pred_future = predict_future_steps(model, init_seq_pos, init_seq_vid, holdout_steps)
        results[pid] = (pred_future, true_future)
        print(f"Person {pid}:")
        print("Predicted Future Positions:")
        print(pred_future)
        print("True Future Positions:")
        print(true_future)
        print("-" * 40)
    return results

if __name__ == '__main__':
    label_file = "Pedestrian_labels/1_frame.txt" 
    session_id = 1
    timesteps = 10
    video_size = 64
    batch_size = 512
    num_epochs = 20

    # Create training dataset and dataloader
    print("Building training dataset...")
    train_dataset = TrajectoryVideoDataset(label_file, session_id, timesteps=timesteps, video_size=video_size)
    print(f"Number of training samples: {len(train_dataset)}")
    print("Building training dataloader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = TrajVideoTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_path = "checkpoint.pth"
    start_epoch = 0 

    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        epoch = start_epoch

    # Training loop
    model.train()
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for pos_seq, vid_seq, target in tqdm(train_loader):
            pos_seq = pos_seq.to(device)
            vid_seq = vid_seq.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(pos_seq, vid_seq).to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * pos_seq.size(0)
        epoch_loss_avg = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/len(train_dataset):.4f}")
            # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss_avg,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Build holdout dictionary from the same label file (or a different one for evaluation)
    holdout_steps = 10
    holdout_dict = build_holdout_dict(label_file, session_id, timesteps=timesteps, holdout_steps=holdout_steps, video_size=video_size)
    
    # Evaluate on selected persons (set selected_persons to None to evaluate all)
    selected_persons = [1, 2]
    eval_results = evaluate_holdout(model, holdout_dict, holdout_steps=holdout_steps, selected_persons=selected_persons)