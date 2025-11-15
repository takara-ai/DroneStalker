from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DroneStalkerModel(nn.Module):
    """
    Unified CNN + Transformer model for drone trajectory prediction.

    Architecture:
    1. CNN Event Feature Encoder: 3-layer CNN processes 64x64 event images -> 128-dim features
    2. Feature Fusion: Kinematic (4-dim) + Event (128-dim) + Positional Encoding (132-dim) -> 264-dim
    3. Transformer Encoder: 4 layers, 4 heads, processes historical sequence
    4. Transformer Decoder: 4 layers, 4 heads, predicts future trajectory
    5. Prediction Head: Maps decoder output to bounding box coordinates

    Input:
        - event_images: [batch, Np, 1, 64, 64] - Sequence of event frames
        - kinematic_features: [batch, Np, 4] - Position and velocity features (z-score normalized)

    Output:
        - predictions: [batch, Nf, 4] - Future bounding boxes (x1, y1, x2, y2) for 18 frames
    """
    def __init__(self, Np=12, Nf=18, input_dim=132, num_layers=4, num_heads=4, dim_feedforward=512, dropout=0.1):
        super(DroneStalkerModel, self).__init__()

        self.Np = Np  # Observation window (12 frames = 0.4s at 30fps)
        self.Nf = Nf  # Prediction horizon (18 frames = 0.6s at 30fps)
        self.input_dim = input_dim  # 132 = 4 kinematic + 128 event features

        # === CNN Event Feature Encoder (Step 2) ===
        # Layer 1: (1, 64, 64) -> (32, 32, 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: (32, 32, 32) -> (64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: (64, 16, 16) -> (128, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layer: Flatten to 128-dim event feature vector
        self.fc_event = nn.Linear(128 * 8 * 8, 128)

        # === Transformer Encoder (Step 4) ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Transformer Decoder (Step 4) ===
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # === Learnable Future Query Tokens ===
        # These represent the 18 future timesteps to be predicted
        self.future_query = nn.Parameter(torch.randn(Nf, input_dim))

        # === Prediction Head ===
        # Maps 132-dim decoder output to 4-dim bounding box (x1, y1, x2, y2)
        self.prediction_head = nn.Linear(input_dim, 4)

        # === Pre-compute Positional Encoding ===
        # Buffer (not a parameter, but part of model state)
        self.register_buffer('positional_encoding',
                           create_sinusoidal_positional_encoding(Np, input_dim))

    def forward(self, event_images, kinematic_features):
        """
        Forward pass: CNN feature extraction + Feature fusion + Transformer prediction.

        Args:
            event_images (torch.Tensor): [batch, Np, 1, 64, 64] - Event frame sequence
            kinematic_features (torch.Tensor): [batch, Np, 4] - Normalized kinematic features

        Returns:
            torch.Tensor: [batch, Nf, 4] - Predicted bounding boxes for future frames
        """
        batch_size, Np, _, _, _ = event_images.shape

        # === Step 1: CNN Event Feature Extraction ===
        # Reshape: [batch, Np, 1, 64, 64] -> [batch*Np, 1, 64, 64]
        images_flat = event_images.view(batch_size * Np, 1, 64, 64)

        # Conv layer 1
        x = self.pool1(F.relu(self.conv1(images_flat)))

        # Conv layer 2
        x = self.pool2(F.relu(self.conv2(x)))

        # Conv layer 3
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten and project to 128-dim event features
        x = x.view(-1, 128 * 8 * 8)
        event_features = self.fc_event(x)  # [batch*Np, 128]

        # Reshape back: [batch*Np, 128] -> [batch, Np, 128]
        event_features = event_features.view(batch_size, Np, 128)

        # === Step 2: Feature Fusion ===
        # Concatenate kinematic + event features: [batch, Np, 4] + [batch, Np, 128] -> [batch, Np, 132]
        fused_features = torch.cat([kinematic_features, event_features], dim=-1)

        # === Step 3: Add Positional Encoding ===
        # positional_encoding: [Np, 132] -> broadcast to [batch, Np, 132]
        pos_enc = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        input_sequence_encoded = fused_features + pos_enc  # [batch, Np, 132]

        # === Step 4: Transformer Encoding ===
        # Process historical sequence through encoder
        encoder_output = self.transformer_encoder(input_sequence_encoded)  # [batch, Np, 132]

        # === Step 5: Transformer Decoding ===
        # Broadcast future query tokens across batch dimension
        future_queries = self.future_query.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, Nf, 132]

        # Decoder predicts future sequence using encoder memory
        decoder_output = self.transformer_decoder(future_queries, encoder_output)  # [batch, Nf, 132]

        # === Step 6: Bounding Box Prediction ===
        # Project to 4-dim bounding box coordinates
        predicted_trajectory = self.prediction_head(decoder_output)  # [batch, Nf, 4]

        return predicted_trajectory


def get_data() -> list[dict]:
    # Load dataset
    raw_data = []
    for x in [0, 1]:
        with open(f"../../data/{x}/coordinates.txt", "r") as coord_file:
            lines = coord_file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                time_part, coords_part = line.split(":", 1)
            except ValueError:
                continue
            time_str = time_part.strip()
            coord_values = [c.strip() for c in coords_part.split(",")]
            if len(coord_values) != 4:
                continue
            try:
                time_val = float(time_str)
                x1, y1, x2, y2 = [float(c) for c in coord_values]
            except ValueError:
                continue
            frame_time_str = str(time_val).replace(".", "")
            img_path = f"../../data/{x}/Event/Frames/Video_{x}_frame_{frame_time_str}.png"
            sample = {
                "time": float(time_val),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "img_path": img_path,
            }
            raw_data.append(sample)
        print(f"Loaded {len(raw_data)} samples from {x}")
    return raw_data

# Get the centre coordinates of the bounding box for one sample
def get_center_coordinates(sample: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]
    xc = (x1 + x2) / 2 # Center x coordinate of bounding box
    yc = (y1 + y2) / 2 # Center y coordinate of bounding box
    return (xc, yc)
    
# Get the velocity for two samples
def get_kinematic_features(sample1: dict, sample2: dict, meanv: tuple[float, float], stdv: tuple[float, float], mean_coordinates: tuple[float, float], std_coordinates: tuple[float, float]) -> tuple[float, float, float, float]:
    x1, y1 = get_center_coordinates(sample1)
    x2, y2 = get_center_coordinates(sample2)
    dx = x2 - x1
    dy = y2 - y1
    dt = sample2["time"] - sample1["time"]
    vx = dx / dt # Velocity in pixels per second (px/s)
    vy = dy / dt # Velocity in pixels per second (px/s)

    # Perform Z-score standardisation
    x_zscore = (x2 - mean_coordinates[0]) / std_coordinates[0]
    y_zscore = (y2 - mean_coordinates[1]) / std_coordinates[1]
    vx_zscore = (vx - meanv[0]) / stdv[0]
    vy_zscore = (vy - meanv[1]) / stdv[1]
    return (x_zscore, y_zscore, vx_zscore, vy_zscore) # Return the x2, y2, vx and vy (z-score standardised)

def get_mean_velocity(data: list[dict]) -> tuple[float, float]:
    """Calculate mean velocity (raw, not z-scored) across the dataset."""
    total_vx = 0
    total_vy = 0
    for i in range(len(data) - 1):
        x1, y1 = get_center_coordinates(data[i])
        x2, y2 = get_center_coordinates(data[i + 1])
        dt = data[i + 1]["time"] - data[i]["time"]
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        total_vx += vx
        total_vy += vy
    mean_vx = total_vx / (len(data) - 1)
    mean_vy = total_vy / (len(data) - 1)
    return (mean_vx, mean_vy)

def get_std_velocity(data: list[dict], mean: tuple[float, float]) -> tuple[float, float]:
    """Calculate standard deviation of velocity (raw, not z-scored) across the dataset."""
    mean_vx, mean_vy = mean
    sum_sq_vx = 0
    sum_sq_vy = 0
    for i in range(len(data) - 1):
        x1, y1 = get_center_coordinates(data[i])
        x2, y2 = get_center_coordinates(data[i + 1])
        dt = data[i + 1]["time"] - data[i]["time"]
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        sum_sq_vx += (vx - mean_vx) ** 2
        sum_sq_vy += (vy - mean_vy) ** 2
    std_vx = math.sqrt(sum_sq_vx / (len(data) - 1))
    std_vy = math.sqrt(sum_sq_vy / (len(data) - 1))
    return (std_vx, std_vy)

def get_mean_coordinates(data: list[dict]) -> tuple[float, float]:
    total_x = 0
    total_y = 0
    for i in range(len(data)):
        x, y = get_center_coordinates(data[i])
        total_x += x
        total_y += y
    mean_x = total_x / len(data)
    mean_y = total_y / len(data)
    return (mean_x, mean_y)

def get_std_coordinates(data: list[dict], mean: tuple[float, float]) -> tuple[float, float]:
    """Calculate standard deviation of center coordinates across the dataset."""
    sum_sq_x = 0
    sum_sq_y = 0
    for i in range(len(data)):
        x, y = get_center_coordinates(data[i])
        sum_sq_x += (x - mean[0]) ** 2
        sum_sq_y += (y - mean[1]) ** 2
    std_x = math.sqrt(sum_sq_x / len(data))
    std_y = math.sqrt(sum_sq_y / len(data))
    return (std_x, std_y)

def get_event_feature(sample: dict) -> torch.Tensor:
    """
    Loads an image from sample["img_path"], crops to bounding box, 
    and rescales to 64x64 pixels, converting to grayscale (single channel).

    Returns:
        torch.Tensor: A 64x64 tensor (grayscale, values in [0,1]).
    """
    img_path = sample["img_path"]
    x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]
    try:
        img = Image.open(img_path)
        cropped_img = img.crop((x1, y1, x2, y2))
        resized_img = cropped_img.resize((64, 64), Image.LANCZOS)
        resized_img = resized_img.convert("L")
        arr = np.array(resized_img, dtype=np.uint8)  # shape: (64, 64)
        tensor = torch.from_numpy(arr).float().div(255.0)  # normalize to [0,1]
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None
    return tensor

def create_sinusoidal_positional_encoding(sequence_length: int, dimension: int) -> torch.Tensor:
    """
    Generates the sinusoidal positional encoding matrix P.
    
    Args:
        sequence_length: Np (e.g., 12) - The length of the observation window.
        dimension: D (e.g., 132) - The feature dimension of the fused input vector (I_t).

    Returns:
        P_matrix (torch.Tensor): The positional encoding matrix of shape (sequence_length, dimension).
    """
    # Initialize P matrix: Shape (Np, D) -> (12, 132)
    P_matrix = torch.zeros(sequence_length, dimension)
    
    # Create the position index tensor (t index: 0, 1, 2, ...)
    # Shape: (Np, 1)
    position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)

    # Create the dimension index tensor (i index: 0, 2, 4, ...)
    # Only even indices are calculated, as they determine the frequency for the pair (i, i+1)
    # i: indices for which we apply sin (0, 2, 4, ..., D-2)
    indices_i = torch.arange(0, dimension, 2, dtype=torch.float)

    # Calculate the denominator (angular frequency term): 1 / (10000^(2i/D))
    # Using log-exp stability trick for powers: exp(2i/D * -log(10000))
    # Shape: (D/2) -> (66)
    div_term = torch.exp(indices_i * (-math.log(10000.0) / dimension))

    # Apply Sine and Cosine functions to fill the matrix P
    # Even indices (0, 2, 4, ...) get sine: P(t, 2i) = sin(t / denominator)
    P_matrix[:, 0::2] = torch.sin(position * div_term)
    
    # Odd indices (1, 3, 5, ...) get cosine: P(t, 2i+1) = cos(t / denominator)
    P_matrix[:, 1::2] = torch.cos(position * div_term)
    
    return P_matrix

def main():
    # Event features

    # Transformer encoder decoder
    pass


if __name__ == "__main__":
    main()

# Help

'''
    # Kinematic features usage

    data = get_data()
    meanv = get_mean_velocity(data)
    stdv = get_std_velocity(data, meanv)
    mean_coordinates = get_mean_coordinates(data)
    std_coordinates = get_std_coordinates(data, mean_coordinates)
    get_kinematic_features(data[0], data[1], meanv, stdv, mean_coordinates, std_coordinates)
'''

'''
Generating and Extracting P_t (Positional Encoding)

    # Constants based on the model:
    D = 132 # Feature dimension (4 kinematic + 128 CNN)
    Np = 12 # Sequence length (0.4s / 33ms)
    # Generate the full positional encoding matrix P for the entire sequence
    P_matrix = create_sinusoidal_positional_encoding(Np, D)
    # Define the time step index you are currently processing (e.g., the last observation)
    t_step = Np - 1 # Let's say we want the PE for the last observed timestep (index 11)

    # Extract the positional encoding vector P_t for the current timestep
    # P_t_raw is (132,)
    P_t_raw = P_matrix[t_step, :] 

    # Reshape P_t to be a 1x132 vector for element-wise addition (fusion)
    P_t = P_t_raw.unsqueeze(0) 
'''

'''
# Input for the transformer encoder decorder
    K_i + EF_i + P_t # Kinematics features, event features (CNN output) and positional encoding input fusion (this is fed to transformer encoder decoder)
'''