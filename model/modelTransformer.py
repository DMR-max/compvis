import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageKeypointsRegressorTransformer(nn.Module):
    def __init__(self, 
                 embed_dim=128, 
                 max_keypoints=100, 
                 num_transformer_layers=2, 
                 num_heads=4,
                 dropout=0.1):
        """
        Args:
            embed_dim: The dimension for the image token and keypoint embeddings.
            max_keypoints: Maximum number of keypoints to expect per image (for padding).
            num_transformer_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads in the transformer.
            dropout: Dropout probability for the transformer layers.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # --- Image Feature Extractor ---
        # A simple CNN that produces a feature map which is flattened and projected
        # to create a single image token.
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),  # further reduces spatial dims
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, H', W')
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # fixed output size (B, 32, 4, 4)
        )
        self.image_fc = nn.Linear(32 * 4 * 4, embed_dim)

        # --- Keypoint Embedding ---
        # Embed each 2D keypoint into an embedding space of size embed_dim.
        self.keypoint_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

        # --- Learnable [CLS] Token ---
        # This token will aggregate information from both image and keypoints.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- Positional Embeddings ---
        # Maximum sequence length: 1 (CLS) + 1 (image token) + max_keypoints.
        self.max_seq_length = 1 + 1 + max_keypoints
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_length, embed_dim))

        # --- Transformer Encoder ---
        # Use batch_first=True so the input shape is (batch_size, seq_length, embed_dim).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- Final Regression Head ---
        self.final_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)  # Output: single value (angle)
        )

    def forward(self, images, keypoints_list):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            keypoints_list: list of length batch_size; each element is a Tensor of shape (K, 2)
                            where K may vary between samples.
        Returns:
            Tensor of shape (batch_size, 1) containing the predicted angles.
        """
        batch_size = images.size(0)

        # 1) Process images to get a single token per image.
        img_feat = self.image_conv(images)  # shape: (batch_size, 32, 4, 4)
        img_feat = img_feat.view(batch_size, -1)  # flatten: (batch_size, 32*4*4)
        img_token = self.image_fc(img_feat)  # shape: (batch_size, embed_dim)
        img_token = img_token.unsqueeze(1)   # shape: (batch_size, 1, embed_dim)

        # 2) Process each sample’s keypoints.
        # For each sample, embed each keypoint using the MLP.
        keypoint_tokens = []
        lengths = []  # actual number of keypoints per sample
        for kp in keypoints_list:
            if kp.size(0) == 0:
                # If no keypoints are provided, use a single zero token.
                token = torch.zeros(1, self.embed_dim, device=images.device)
                length = 1
            else:
                token = self.keypoint_mlp(kp)  # shape: (K, embed_dim)
                length = token.size(0)
            keypoint_tokens.append(token)
            lengths.append(length)
        
        # 3) Pad keypoint tokens to the maximum length in the batch.
        max_k = max(lengths)
        padded_keypoints = []
        for token in keypoint_tokens:
            pad_size = max_k - token.size(0)
            if pad_size > 0:
                pad = torch.zeros(pad_size, self.embed_dim, device=images.device)
                token = torch.cat([token, pad], dim=0)
            padded_keypoints.append(token)
        # Stack into a batch: shape (batch_size, max_k, embed_dim)
        keypoint_tokens_batch = torch.stack(padded_keypoints, dim=0)

        # 4) Construct the token sequence for each sample:
        # Sequence = [CLS] token, image token, keypoint tokens.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: (batch_size, 1, embed_dim)
        sequence = torch.cat([cls_tokens, img_token, keypoint_tokens_batch], dim=1)
        # The sequence length is now: 1 (CLS) + 1 (image) + max_k (keypoints)
        seq_length = sequence.size(1)

        # 5) Add positional embeddings.
        pos_emb = self.pos_embedding[:, :seq_length, :]
        sequence = sequence + pos_emb

        # 6) Create an attention mask for padded keypoint positions.
        # The first two tokens (CLS and image) are always valid.
        attn_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=images.device)
        for i, l in enumerate(lengths):
            # Keypoint tokens start at index 2.
            if l < max_k:
                attn_mask[i, 2 + l:] = True  # Mask padded positions

        # 7) Pass the sequence through the transformer encoder.
        transformer_out = self.transformer(sequence, src_key_padding_mask=attn_mask)
        # Use the output corresponding to the CLS token (position 0).
        cls_out = transformer_out[:, 0, :]

        # 8) Predict the angle from the CLS token output.
        angle_out = self.final_fc(cls_out)  # shape: (batch_size, 1)
        # Optional: if you need to constrain the output to a specific range (e.g., [0, 360)),
        # you can apply a sigmoid and scale accordingly.
        # angle_out = torch.sigmoid(angle_out) * 360.0
        
        return angle_out
