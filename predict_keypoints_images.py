import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from torchvision import transforms, models
from PIL import Image

###############################################################################
# 1. CONFIGURATION
###############################################################################
FILTERED_DATA_PKL = "C:/Users/Bulut/Documents/GitHub/compvis/Filtered_data_list.pkl"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Example transforms for images (resize to 224x224, convert to tensor)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # If using pretrained networks, you often do:
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])

BATCH_SIZE    = 8
TRAIN_SPLIT   = 0.8
EPOCHS        = 10
LEARNING_RATE = 1e-3

print(f"Using device: {DEVICE}")

###############################################################################
# 2. DATASET & DATALOADER
###############################################################################

class ImageKeypointDataset(Dataset):
    """
    Dataset that provides:
      1) Image,
      2) Raw keypoint coordinates,
      3) Label (angle of movement).
    """
    def __init__(self, data_list, transform=None):
        """
        data_list: a list of tuples -> (image_path, keypoints, angle)
        transform: torchvision transforms for images
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, keypoints, angle = self.data_list[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert keypoints to tensor (seq_len, 2)
        kp_tensor = torch.tensor(keypoints, dtype=torch.float32)

        # Convert angle to float tensor, shape (1,)
        angle_tensor = torch.tensor(angle, dtype=torch.float32).unsqueeze(0)

        return image, kp_tensor, angle_tensor


def multimodal_collate_fn(batch):
    """
    Custom collate function to handle:
      - a list of (image, keypoints, label)
      - variable-length keypoints
      - images get stacked
    """
    images, kpoints_list, labels = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)  # (batch_size, C, H, W)

    # Pad keypoints to the same sequence length
    padded_kpoints = pad_sequence(kpoints_list, batch_first=True, padding_value=0.0)

    # Stack labels
    labels = torch.stack(labels, dim=0)  # (batch_size, 1)

    return images, padded_kpoints, labels

###############################################################################
# 3. MULTI-MODAL MODEL
###############################################################################

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True, out_features=128):
        super().__init__()
        # Use a pretrained ResNet18 as an example
        backbone = models.resnet18(pretrained=pretrained)
        # Remove final classification layer
        num_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        
        self.backbone = backbone
        self.projection = nn.Linear(num_feats, out_features)

    def forward(self, x):
        """
        x shape: (batch_size, 3, H, W)
        """
        features = self.backbone(x)         # (batch_size, 512) for ResNet18
        out = self.projection(features)     # (batch_size, out_features)
        return out


class KeypointEncoder(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # Note: remove batch_first here
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
            # batch_first=False is default
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, 2]
        """
        # 1) Embed => [batch_size, seq_len, model_dim]
        x = self.embedding(x)

        # 2) Transpose => [seq_len, batch_size, model_dim]
        x = x.transpose(0, 1)

        # 3) Pass through transformer => still [seq_len, batch_size, model_dim]
        x = self.transformer_encoder(x)

        # 4) Transpose back => [batch_size, seq_len, model_dim]
        x = x.transpose(0, 1)

        # 5) Pool across seq_len => [batch_size, model_dim]
        x = x.mean(dim=1)

        # 6) Final FC => [batch_size, model_dim]
        x = self.fc_out(x)

        return x


class MultiModalRegressor(nn.Module):
    def __init__(self, img_out_features=128, keypoint_dim=64, hidden_dim=128):
        super().__init__()
        self.img_encoder = ImageEncoder(pretrained=True, out_features=img_out_features)
        self.kp_encoder  = KeypointEncoder(input_dim=2, model_dim=keypoint_dim)

        fusion_input_dim = img_out_features + keypoint_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # single angle
        )

    def forward(self, images, keypoints):
        # Encode image
        img_feats = self.img_encoder(images)   # (batch_size, img_out_features)

        # Encode keypoints
        kp_feats  = self.kp_encoder(keypoints) # (batch_size, keypoint_dim)

        # Fuse
        fused = torch.cat([img_feats, kp_feats], dim=1)  # (batch_size, fusion_input_dim)

        # Regress angle
        out = self.regressor(fused)  # (batch_size, 1)
        return out


###############################################################################
# 4. TRAINING & TESTING FUNCTIONS
###############################################################################

def train_model(model, train_loader, epochs=10, lr=1e-3):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, kpoints, labels in train_loader:
            images = images.to(DEVICE)
            kpoints = kpoints.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(images, kpoints)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def test_model(model, test_loader):
    model.eval()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, kpoints, labels in test_loader:
            images = images.to(DEVICE)
            kpoints = kpoints.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(images, kpoints)
            loss = criterion(preds, labels)
            total_loss += loss.item()

            # If you want to visualize some predictions:
            # print("Predicted:", preds.squeeze().tolist())
            # print("Actual:   ", labels.squeeze().tolist())

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")


###############################################################################
# 5. MAIN EXECUTION
###############################################################################

def main():
    # 1) Load filtered data
    with open(FILTERED_DATA_PKL, 'rb') as f:
        filtered_data_list = pickle.load(f)

    print(f"Loaded filtered data list, total samples: {len(filtered_data_list)}")

    # 2) Create dataset
    dataset = ImageKeypointDataset(filtered_data_list, transform=IMAGE_TRANSFORM)

    # 3) Train/Test split
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 4) Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=multimodal_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=multimodal_collate_fn
    )

    # 5) Initialize multi-modal model
    model = MultiModalRegressor(
        img_out_features=128,
        keypoint_dim=64,
        hidden_dim=128
    )
    print(model)

    # 6) Train
    print("Starting Training ...")
    train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    # 7) Test
    print("Starting Testing ...")
    test_model(model, test_loader)


if __name__ == "__main__":
    main()