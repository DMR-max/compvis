import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

###############################################################################
# 1. CONFIGURATION
###############################################################################
#DATA_PKL      = "C:/Users/Bulut/Documents/GitHub/compvis/images.pkl"  # <-- Replace with your path
DATA_PKL      = os.path.join(os.getcwd(), "images.pkl")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 5
EPOCHS        = 1
LEARNING_RATE = 0.001
TRAIN_SPLIT   = 0.8  # Use 90% of the dataset for training and 10% for testing

# Basic image transforms
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    # If using a pretrained ResNet, you typically want normalization:
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"Using device: {DEVICE}")

###############################################################################
# 2. DATASET
###############################################################################
class ImageAngleDataset(Dataset):
    """
    Dataset that returns:
      - image tensor
      - angle (float) as a label
    """
    def __init__(self, data_list, transform=None):
        """
        data_list: List of (image_path, angle)
        transform: TorchVision transforms for images
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, angle = self.data_list[idx]
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert angle to float tensor [1,]
        angle_tensor = torch.tensor(angle, dtype=torch.float32).unsqueeze(0)
        return image, angle_tensor

###############################################################################
# 3. MODEL: Simple ResNet-based regressor
###############################################################################
# class ImageRegressor(nn.Module):
#     def __init__(self, pretrained=True):
#         """
#         If pretrained=True, uses pretrained ImageNet weights.
#         If pretrained=False, initializes from scratch.
#         """
#         super().__init__()
#         # Use a ResNet18 as the backbone
#         if pretrained:
#             backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         else:
#             backbone = models.resnet18(weights=None)

#         # Remove the final classification layer
#         num_feats = backbone.fc.in_features
#         backbone.fc = nn.Identity()

#         self.backbone = backbone
#         # Final linear to produce 1 output (angle)
#         self.fc = nn.Linear(num_feats, 1)

#     def forward(self, x):
#         # x: [batch_size, 3, H, W]
#         features = self.backbone(x)   # [batch_size, 512] for ResNet18
#         out = self.fc(features)       # [batch_size, 1]
#         return out

###############################################################################
# 3. MODEL: Simple ResNet-based regressor
###############################################################################
class ImageRegressor(nn.Module):
    def __init__(self, pretrained=True):
        """
        If pretrained=True, uses pretrained ImageNet weights.
        If pretrained=False, initializes from scratch.
        """
        super().__init__()
        # Use a ResNet18 as the backbone
        if pretrained:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=None)

        # Remove the final classification layer
        num_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        # Final linear to produce 1 output (angle)
        #       But with an extra layer in between to smoothen the process
        self.fc_before = nn.Linear(num_feats, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: [batch_size, 3, H, W]
        features = self.backbone(x)   # [batch_size, 512] for ResNet18
        before_out = self.fc_before(features)
        out = self.fc(before_out)       # [batch_size, 1]
        return out

###############################################################################
# 4. TRAINING & TESTING
###############################################################################
def circular_error(pred, actual):
    """
    Computes the circular error (in degrees) between predicted and actual angles.
    The error is defined as the minimum of the absolute difference and 360 minus that difference.
    """
    diff = abs(pred - actual)
    return diff if diff <= 180 else 360 - diff

def train_model(model, train_loader, epochs=10, lr=1e-3):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, angles in train_loader:
            images = images.to(DEVICE)
            angles = angles.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, angles)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_loss:.4f}")

def test_model(model, test_loader):
    model.to(DEVICE)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    all_preds = []
    all_labels = []
    total_circular_error = 0.0
    count = 0

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(DEVICE)
            angles = angles.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, angles)
            total_loss += loss.item()

            preds_list = preds.cpu().view(-1).tolist()
            angles_list = angles.cpu().view(-1).tolist()
            all_preds.extend(preds_list)
            all_labels.extend(angles_list)

            for p, a in zip(preds_list, angles_list):
                err = circular_error(p, a)
                total_circular_error += err
                count += 1

    avg_loss = total_loss / len(test_loader)
    avg_circular_error = total_circular_error / count if count > 0 else 0.0

    print(f"Test Loss: {avg_loss:.4f}")
    print("Sample Predictions vs Actual with Circular Error:")
    for i in range(len(all_preds)):
        err = circular_error(all_preds[i], all_labels[i])
        print(f"  Pred: {all_preds[i]:.2f}, Actual: {all_labels[i]:.2f}, Circular Error: {err:.2f}")
    print(f"Average Circular Error: {avg_circular_error:.2f}")
    

def inference(model, video, batch_size=128, frame_size=1):
    #Video shape = (batch_size, 10frames, 3, 128, 128)
    #Resize image to (batch_size, 10frames, 3, 244, 244)
    print(video.shape)
    #video = video.permute(0, 1, 4, 2, 3)
    print(video.shape)
    video = video[:, 0, :, :, :]
    #video = video.view(batch_size * frame_size, 3, 128, 128)
    #print(video)
    #video = IMAGE_TRANSFORM(video_reshaped)
    video = F.interpolate(video, size=(244, 244), mode='bilinear', align_corners=False).to(DEVICE)
    
    
    
    model.eval()
    #image = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    with torch.no_grad():
        pred = model(video)
    #pred_angle = pred.view(batch_size, frame_size, -1)
    pred_angle = pred.unsqueeze(1).repeat(1, 10, 1)
    print("ANGLE")
    print(pred.shape)
    print(pred_angle.shape)
    return pred_angle

###############################################################################
# 5. MAIN
###############################################################################
def main():
    # 1) Load data_list from .pkl
    with open(DATA_PKL, "rb") as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} samples from {DATA_PKL}")

    # 2) Create dataset
    dataset = ImageAngleDataset(data_list, transform=IMAGE_TRANSFORM)

    # 3) Split into train/test
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # 4) Initialize model
    model = ImageRegressor(pretrained=True)
    print(model)

    # 5) Train
    print("Starting Training ...")
    train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    # 6) Test with detailed circular error analysis
    print("Starting Testing ...")
    test_model(model, test_loader)
    
    return model


if __name__ == "__main__":
    main()
