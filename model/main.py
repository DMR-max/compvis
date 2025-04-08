import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import pickle
import filtering
import modelTransformer
import training

def circular_error(pred, actual):
    """
    Computes the circular error (in degrees) between predicted and actual angles.
    The error is defined as the minimum of the absolute difference and 360 minus that difference.
    """
    diff = abs(pred - actual)
    return diff if diff <= 180 else 360 - diff

def test_model(model, test_loader, device):
    model.eval()
    errors = []
    print("\nTest Results:")
    with torch.no_grad():
        for images, keypoints_list, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)
            outputs = model(images, [kp.to(device) for kp in keypoints_list])
            outputs = outputs.squeeze(-1)  # shape (batch,)
            targets = angles.squeeze(-1)
            
            # Process each sample in the batch
            for pred, actual in zip(outputs, targets):
                pred_val = pred.item()
                actual_val = actual.item()
                err = circular_error(pred_val, actual_val)
                errors.append(err)
                print(f"Actual: {actual_val:.2f}, Predicted: {pred_val:.2f}, Circular Error: {err:.2f}")
                
    avg_error = sum(errors) / len(errors) if errors else 0
    print(f"\nAverage Circular Error: {avg_error:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_json', type=str, default='data.json',
                        help='Path to JSON file with records of {img_path, keypoints, angle}')
    parser.add_argument('--epochs',    type=int, default=15)
    parser.add_argument('--batch_size',type=int, default=4)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default='checkpoint_multi_kpts.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imagesPath = "C:/Users/Bulut/Documents/GitHub/compvis/Dataset/Images/" 
    directory = "C:/Users/Bulut/Documents/GitHub/compvis/"

    # 1) Load data_list from pickle file
    with open(directory + 'final.pkl', 'rb') as f:
        data_dict = pickle.load(f)  # Assuming it's stored as a list of records

    # Use 90% of data for training and 10% for testing
    split_idx = int(0.8 * len(data_dict))
    train_list = data_dict[:split_idx]
    test_list  = data_dict[split_idx:]


    # 2) Create Dataset & DataLoader objects
    train_dataset = filtering.ImageKeypointsDataset(train_list)
    test_dataset  = filtering.ImageKeypointsDataset(test_list)

    train_loader = filtering.DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=filtering.collate_fn_varlen)
    test_loader   = filtering.DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4,
                              collate_fn=filtering.collate_fn_varlen)
    
    # 3) Create model
    model = modelTransformer.ImageKeypointsRegressorTransformer(
        embed_dim=128,         # Embedding dimension for both image and keypoints
        max_keypoints=100,     # Maximum number of keypoints (adjust as needed)
        num_transformer_layers=2,
        num_heads=4,
        dropout=0.1
    ).to(device)

    # 4) Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5) Training loop
    best_test_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = training.train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss  = training.evaluate(model, test_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Save best checkpoint based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Test loss improved to {best_test_loss:.4f}, saved model.")

    print("Training complete.")

    # 6) Evaluate on the held-out test set with detailed error analysis
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
