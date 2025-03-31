import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ImageKeypointsDataset(Dataset):
    """
    Each item has:
       - An image path
       - A list of (x,y) keypoints
       - A single angle label
    """
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list: a list of dicts, each like:
                {
                  "img_path": str,
                  "keypoints": List[List[float]], # shape (K, 2)
                  "angle": float
                }
            transform: optional function to apply to the image
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        record = self.data_list[idx]
        img_path = record[0]
        keypoints = record[1]  # list of [x, y]
        angle = record[2]

        # --- Load Image ---
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        
        desired_size = (151, 385)
        image = cv2.resize(image, desired_size, interpolation=cv2.INTER_AREA)

        if self.transform:
            # If you're using torchvision transforms, you might first convert NumPy->PIL Image
            # or adapt the transform for NumPy arrays
            image = self.transform(image)
        else:
            # Minimal transform: scale to [0,1] and CHW
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))  # (C, H, W)

        # Convert keypoints to a float tensor shape (K, 2)
        keypoints_arr = np.array(keypoints, dtype=np.float32)  # shape (K,2)

        # Single angle -> shape (1,)
        angle_arr = np.array([angle], dtype=np.float32)

        # Convert to torch tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)
        keypoints_tensor = torch.tensor(keypoints_arr, dtype=torch.float32)
        angle_tensor = torch.tensor(angle_arr, dtype=torch.float32)

        return image_tensor, keypoints_tensor, angle_tensor


def collate_fn_varlen(batch):
    """
    Custom collate function to handle variable-length keypoints.
    batch: list of (image_tensor, keypoints_tensor, angle_tensor) for each item
    We stack images and angles, but keep keypoints in a list of tensors.
    """
    images  = []
    keypoints_list = []
    angles  = []
    for (img, kpts, ang) in batch:
        images.append(img)
        keypoints_list.append(kpts)
        angles.append(ang)

    # Stack images => (batch_size, C, H, W)
    images = torch.stack(images, dim=0)
    # angles => (batch_size, 1)
    angles = torch.stack(angles, dim=0)

    # keypoints_list is left as a python list of shape (K_i, 2)
    # Because each sample can have different K_i
    # We'll handle it in the model or training loop
    return images, keypoints_list, angles



