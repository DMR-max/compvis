import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset from .pkl file
directory = "C:/Users/Bulut/Documents/GitHub/compvis/"  # Update this to the correct path
with open(directory + 'finalKeypoints.pkl', 'rb') as f:
    data_dict = pickle.load(f)  # Assuming it's stored as a dictionary

# Flatten data_dict to ensure each data sample is a 2D tensor of shape (seq_len, 2)
data = []
labels = []
for item in data_dict:
    # Convert each sequence into a tensor of shape (seq_len, 2) and each label as a float tensor
    sequence = torch.tensor(item[0], dtype=torch.float32)  # Ensure this is a tensor of shape (seq_len, 2)
    label = torch.tensor(item[1], dtype=torch.float32).unsqueeze(0)   # Labels are the angle (single value)
    data.append(sequence)
    labels.append(label)

# Padding function for variable-length sequences
def pad_batch(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # Pad with zeros
    return padded_sequences, torch.stack(labels)

# Dataset Class
class CoordDataset(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.tensor(seq, dtype=torch.float32) for seq in data]  # Convert to tensor
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Convert to tensor and reshape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # Tensor of shape (seq_len, 2)
        y = self.labels[idx]  # Keep as a single value (not sin/cos)
        return x, y

# Split dataset into training and testing (90% training, 10% testing)
dataset = CoordDataset(data, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_batch)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=pad_batch)

# Transformer Model
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, 
                                                        dim_feedforward=ff_dim, dropout=dropout, 
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, 1)  # Output single angle value

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, model_dim)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, model_dim)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc_out(x)  # (batch_size, 1)

# Circular error function
def circular_error(pred, actual):
    """
    Computes the circular error (in degrees) between predicted and actual angles.
    The error is the minimum of the absolute difference and 360 minus that difference.
    """
    diff = abs(pred - actual)
    return diff if diff <= 180 else 360 - diff

# Training Function
def train_model(model, train_dataloader, epochs=15, lr=0.001):
    model.to(device)  # Move the model to GPU (if available)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)  # Move data to GPU
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader):.4f}")

# Testing Function
def test_model(model, test_dataloader):
    model.to(device)  # Ensure the model is on the correct device
    model.eval()
    total_loss = 0
    total_circular_error = 0
    criterion = nn.MSELoss()
    count = 0
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)  # Move data to GPU
            predictions = model(x)
            loss = criterion(predictions, y)
            total_loss += loss.item()
            
            # Compute and print circular error for each sample in the batch
            preds = predictions.squeeze()
            actuals = y.squeeze()
            # Ensure both preds and actuals are iterable
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if actuals.dim() == 0:
                actuals = actuals.unsqueeze(0)
            for pred, actual in zip(preds.tolist(), actuals.tolist()):
                err = circular_error(pred, actual)
                total_circular_error += err
                count += 1
                print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}, Circular Error: {err:.2f}")
    
    avg_loss = total_loss / len(test_dataloader)
    avg_circular_error = total_circular_error / count if count > 0 else 0
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Average Circular Error: {avg_circular_error:.2f}")

# Train the model
model = TransformerRegressor()
train_model(model, train_dataloader)

# Test the model with detailed circular error analysis
test_model(model, test_dataloader)
