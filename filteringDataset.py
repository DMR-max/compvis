import pickle
import torch

import os
from PIL import Image

# Adjust paths as needed
pkl_path = "C:/Users/Bulut/Documents/GitHub/compvis/Dataset/CombinedData.pkl"
images_dir = "C:/Users/Bulut/Documents/GitHub/compvis/Images/"
output_pkl_path = "C:/Users/Bulut/Documents/GitHub/compvis/images.pkl"

# Load your CombinedData.pkl
with open(pkl_path, 'rb') as f:
    data_dict = pickle.load(f)  # Suppose it's a list of [keypoints, angle]
    
#print(f"Loaded {len(data_dict)} samples from {pkl_path}")
#print(f"Example data: {data_dict[0][0]}")
#print(f"Example data: {data_dict[0][1]}")





filtered_data_list = []

for i, item in enumerate(data_dict):
    keypoints = item[0]
    angle = torch.tensor(item[1], dtype=torch.float32).unsqueeze(0)   # Labels are the angle (single value)

    # Convert to torch tensors for checking shape (or you can just check Python lists)
    sequence = torch.tensor(keypoints, dtype=torch.float32)

    if sequence.ndimension() == 3 and sequence.shape[0] == 1:
        sequence = sequence.squeeze(0)  # Remove the first singleton dimension

    # Check if sequence is empty or has invalid shape
    # Example: we want sequence of shape (seq_len, 2) and seq_len>0
    if sequence.shape[0] == 0 or sequence.shape[1] != 2:
        # Skip this row
        continue
    
    # If we reach here, the row is valid
    # The corresponding image is: "image_{i}.jpg" or some pattern
    image_path = os.path.join(images_dir, f"image_{i}.jpg")

    # (Optionally) check if the image actually exists on disk
    if not os.path.isfile(image_path):
        print(f"Warning: Image not found at {image_path}, skipping.")
        continue

    # Keep the data
    filtered_data_list.append((image_path, angle))

# Now we have a filtered list of (image_path, keypoints, angle)
#print(f"Total valid samples: {len(filtered_data_list)}")
#print(f"Example valid sample: {filtered_data_list[135][0]}")
#print(f"Example valid sample: {filtered_data_list[135][1]}")
#print(f"Example valid sample: {filtered_data_list[135][2]}")
#
#print(f"Filtered data type : {type(filtered_data_list)}")
#print(f"Filtered data type0 : {type(filtered_data_list[0][0])}")
#print(f"Filtered data type1 : {type(filtered_data_list[0][1])}")
#print(f"Filtered data type2 : {type(filtered_data_list[0][2  ])}")












 #If desired, save the filtered data to a new pkl
with open(output_pkl_path, 'wb') as f:
    pickle.dump(filtered_data_list, f)


print(f"Filtered data saved to: {output_pkl_path}")