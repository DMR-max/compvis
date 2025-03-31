import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, keypoints_list, angles in dataloader:
        images = images.to(device)   # (batch, 3, H, W)
        angles = angles.to(device)   # (batch, 1)
        # keypoints_list is a list of (K_i, 2) on CPU; we can move each to device in forward()

        optimizer.zero_grad()
        outputs = model(images, [kp.to(device) for kp in keypoints_list])  # (batch, 1)
        outputs = outputs.squeeze(-1)  # (batch,)
        targets = angles.squeeze(-1)   # (batch,)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, keypoints_list, angles in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            outputs = model(images, [kp.to(device) for kp in keypoints_list])
            outputs = outputs.squeeze(-1)
            targets = angles.squeeze(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)
