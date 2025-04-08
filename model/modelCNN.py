import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageKeypointsRegressor(nn.Module):
    def __init__(self, 
                 keypoint_dim=2,          # each keypoint is (x,y)
                 keypoint_embed_dim=16,   # dimension of each embedded keypoint
                 cnn_out_dim=128,         # dimension of CNN features
                 final_hidden_dim=64):
        super().__init__()

        # --- CNN feature extractor (simple example) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))  # final 4×4 feature map
        )
        self.cnn_fc = nn.Linear(32 * 4 * 4, cnn_out_dim)

        # --- MLP for a single keypoint ---
        self.keypoint_mlp = nn.Sequential(
            nn.Linear(keypoint_dim, 32),
            nn.ReLU(),
            nn.Linear(32, keypoint_embed_dim)
        )

        # --- Final regressor ---
        combined_dim = cnn_out_dim + keypoint_embed_dim
        self.final_fc = nn.Sequential(
            nn.Linear(combined_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Linear(final_hidden_dim, 1)  # single value (angle)
        )

    def forward(self, images, keypoints_list):
        """
        images: (batch_size, 3, H, W)
        keypoints_list: a list of length batch_size, 
                        each element is a Tensor (K_i, 2)
                        (variable K_i per sample).
        Returns: shape (batch_size, 1) => predicted angle
        """

        batch_size = images.shape[0]

        # 1) CNN on images
        feats = self.cnn(images)                     # (batch, 32, 4, 4)
        feats = feats.view(batch_size, -1)           # (batch, 32*4*4)
        feats = self.cnn_fc(feats)                   # (batch, cnn_out_dim)

        # 2) Keypoint embedding
        # We'll embed each sample's keypoints and then average pool (or another aggregator)
        kpts_embs = []
        for i in range(batch_size):
            kpts = keypoints_list[i]  # shape (K_i, 2)
            if kpts.size(0) == 0:
                # no keypoints? handle gracefully, e.g., zero vector
                emb_avg = torch.zeros((1, ), device=images.device)
                emb_avg = emb_avg.expand(self.keypoint_mlp[-1].out_features)
            else:
                # MLP each keypoint
                emb = self.keypoint_mlp(kpts)  # (K_i, keypoint_embed_dim)
                # average pool across K_i
                emb_avg = emb.mean(dim=0)      # (keypoint_embed_dim,)
            kpts_embs.append(emb_avg)
        # stack => (batch_size, keypoint_embed_dim)
        kpts_embs = torch.stack(kpts_embs, dim=0)

        # 3) Combine image feats + keypoints
        combined = torch.cat([feats, kpts_embs], dim=1)  # (batch, cnn_out_dim + keypoint_embed_dim)

        # 4) Regress final angle
        angle_out = self.final_fc(combined)  # (batch, 1)

        # If you strictly need [0,360), you can do:
        # angle_out = torch.sigmoid(angle_out) * 360.0
        return angle_out
