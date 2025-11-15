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
        - predictions: [batch, Nf, 4] - Future bounding boxes (x1, y1, x2, y2) for 12 frames
    """
    def __init__(self, Np=12, Nf=12, input_dim=132, num_layers=4, num_heads=4, dim_feedforward=512, dropout=0.1):
        super(DroneStalkerModel, self).__init__()

        self.Np = Np  # Observation window (12 frames = 0.4s at 30fps)
        self.Nf = Nf  # Prediction horizon (12 frames = 0.4s at 30fps)
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

        # === Learnable Query Tokens ===
        # Past reconstruction queries (Np frames)
        self.past_query = nn.Parameter(torch.randn(Np, input_dim))
        # Future prediction queries (Nf frames)
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
        Forward pass: CNN feature extraction + Feature fusion + Transformer prediction + Reconstruction.

        Args:
            event_images (torch.Tensor): [batch, Np, 1, 64, 64] - Event frame sequence
            kinematic_features (torch.Tensor): [batch, Np, 4] - Normalized kinematic features

        Returns:
            tuple: (reconstructed_past, predicted_future)
                - reconstructed_past: [batch, Np, 4] - Reconstructed bounding boxes for observation window
                - predicted_future: [batch, Nf, 4] - Predicted bounding boxes for future frames
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

        # === Step 5a: Reconstruction (Decode Past) ===
        # Broadcast past query tokens across batch dimension
        past_queries = self.past_query.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, Np, 132]

        # Decoder reconstructs observation window using encoder memory
        decoder_output_past = self.transformer_decoder(past_queries, encoder_output)  # [batch, Np, 132]

        # Project to 4-dim bounding box coordinates
        reconstructed_past = self.prediction_head(decoder_output_past)  # [batch, Np, 4]

        # === Step 5b: Prediction (Decode Future) ===
        # Broadcast future query tokens across batch dimension
        future_queries = self.future_query.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, Nf, 132]

        # Decoder predicts future sequence using encoder memory
        decoder_output_future = self.transformer_decoder(future_queries, encoder_output)  # [batch, Nf, 132]

        # Project to 4-dim bounding box coordinates
        predicted_future = self.prediction_head(decoder_output_future)  # [batch, Nf, 4]

        return reconstructed_past, predicted_future


class FREDLoss(nn.Module):
    """
    Custom loss function for FRED trajectory forecasting.

    L = L_Nf + λ * L_Np

    Where:
    - L_Nf: Forecasting loss (L2 distance for future predictions)
    - L_Np: Reconstruction loss (L2 distance for past reconstruction)
    - λ: Scaling coefficient (0.5 in FRED benchmark)
    """
    def __init__(self, lambda_recon=0.5):
        super(FREDLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.mse_loss = nn.MSELoss()

    def forward(self, reconstructed_past, predicted_future, ground_truth_past, ground_truth_future):
        """
        Compute the total FRED loss.

        Args:
            reconstructed_past: [batch, Np, 4] - Model's reconstruction of observation window
            predicted_future: [batch, Nf, 4] - Model's prediction of future frames
            ground_truth_past: [batch, Np, 4] - Ground truth bounding boxes for observation window
            ground_truth_future: [batch, Nf, 4] - Ground truth bounding boxes for future frames

        Returns:
            tuple: (total_loss, forecasting_loss, reconstruction_loss)
        """
        # L_Nf: Forecasting loss (L2 distance / RMSE)
        # MSE = mean squared error, sqrt(MSE) = RMSE = L2 distance
        forecasting_loss = torch.sqrt(self.mse_loss(predicted_future, ground_truth_future))

        # L_Np: Reconstruction loss (L2 distance / RMSE)
        reconstruction_loss = torch.sqrt(self.mse_loss(reconstructed_past, ground_truth_past))

        # Total loss: L = L_Nf + λ * L_Np
        total_loss = forecasting_loss + self.lambda_recon * reconstruction_loss

        return total_loss, forecasting_loss, reconstruction_loss


def get_data() -> list[dict]:
    # Load dataset
    raw_data = []
    for x in range(11):
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
    dt = max((sample2["time"] - sample1["time"]), 0.0001)
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
        dt = max((data[i + 1]["time"] - data[i]["time"]), 0.0001)
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
        dt = max((data[i + 1]["time"] - data[i]["time"]), 0.0001)
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


class DroneTrajectoryDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for FRED drone trajectory prediction with reconstruction.

    Creates sliding windows of Np+Nf consecutive frames.
    Returns:
    - Observation window (Np frames): event images + kinematic features + ground truth past boxes
    - Prediction target (Nf frames): ground truth future boxes
    """
    def __init__(self, data, Np=12, Nf=12):
        self.data = data
        self.Np = Np
        self.Nf = Nf
        self.sequence_length = Np + Nf

        # Compute normalization statistics
        print("Computing normalization statistics...")
        self.mean_velocity = get_mean_velocity(data)
        self.std_velocity = get_std_velocity(data, self.mean_velocity)
        self.mean_coordinates = get_mean_coordinates(data)
        self.std_coordinates = get_std_coordinates(data, self.mean_coordinates)

        # Find valid sequence indices
        self.valid_indices = []
        for i in range(len(data) - self.sequence_length + 1):
            is_valid = True
            for j in range(i, i + self.sequence_length - 1):
                time_diff = data[j + 1]["time"] - data[j]["time"]
                if time_diff > 0.1:  # 100ms gap indicates missing frames
                    is_valid = False
                    break
            if is_valid:
                self.valid_indices.append(i)

        print(f"Created {len(self.valid_indices)} valid sequences from {len(data)} frames")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Extract observation window (Np frames) and future frames (Nf frames)
        obs_frames = self.data[start_idx : start_idx + self.Np]
        future_frames = self.data[start_idx + self.Np : start_idx + self.sequence_length]

        # === Event Images ===
        event_images = []
        for sample in obs_frames:
            event_tensor = get_event_feature(sample)
            if event_tensor is None:
                event_tensor = torch.zeros(64, 64)
            event_images.append(event_tensor.unsqueeze(0))
        event_images = torch.stack(event_images)  # [Np, 1, 64, 64]

        # === Kinematic Features ===
        kinematic_features = []
        for i in range(len(obs_frames) - 1):
            features = get_kinematic_features(
                obs_frames[i], obs_frames[i + 1],
                self.mean_velocity, self.std_velocity,
                self.mean_coordinates, self.std_coordinates
            )
            kinematic_features.append(features)
        # Duplicate last velocity for final frame
        kinematic_features.append(kinematic_features[-1])
        kinematic_features = torch.tensor(kinematic_features, dtype=torch.float32)  # [Np, 4]

        # === Ground Truth Past (for reconstruction loss) ===
        ground_truth_past = []
        for sample in obs_frames:
            bbox = [sample["x1"], sample["y1"], sample["x2"], sample["y2"]]
            ground_truth_past.append(bbox)
        ground_truth_past = torch.tensor(ground_truth_past, dtype=torch.float32)  # [Np, 4]

        # === Ground Truth Future (for forecasting loss) ===
        ground_truth_future = []
        for sample in future_frames:
            bbox = [sample["x1"], sample["y1"], sample["x2"], sample["y2"]]
            ground_truth_future.append(bbox)
        ground_truth_future = torch.tensor(ground_truth_future, dtype=torch.float32)  # [Nf, 4]

        return event_images, kinematic_features, ground_truth_past, ground_truth_future


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda'):
    """
    Training loop for DroneStalkerModel with FRED loss (forecasting + reconstruction).
    """
    model = model.to(device)
    criterion = FREDLoss(lambda_recon=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_forecast_loss = 0.0
        train_recon_loss = 0.0

        for batch_idx, (event_images, kinematic_features, gt_past, gt_future) in enumerate(train_loader):
            event_images = event_images.to(device)
            kinematic_features = kinematic_features.to(device)
            gt_past = gt_past.to(device)
            gt_future = gt_future.to(device)

            # Forward pass
            reconstructed_past, predicted_future = model(event_images, kinematic_features)

            # Compute loss
            loss, forecast_loss, recon_loss = criterion(reconstructed_past, predicted_future, gt_past, gt_future)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_forecast_loss += forecast_loss.item()
            train_recon_loss += recon_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Forecast: {forecast_loss.item():.4f}, Recon: {recon_loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_forecast = train_forecast_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_forecast_loss = 0.0
        val_recon_loss = 0.0

        with torch.no_grad():
            for event_images, kinematic_features, gt_past, gt_future in val_loader:
                event_images = event_images.to(device)
                kinematic_features = kinematic_features.to(device)
                gt_past = gt_past.to(device)
                gt_future = gt_future.to(device)

                reconstructed_past, predicted_future = model(event_images, kinematic_features)
                loss, forecast_loss, recon_loss = criterion(reconstructed_past, predicted_future, gt_past, gt_future)

                val_loss += loss.item()
                val_forecast_loss += forecast_loss.item()
                val_recon_loss += recon_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_forecast = val_forecast_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - Total: {avg_train_loss:.4f}, Forecast: {avg_train_forecast:.4f}, Recon: {avg_train_recon:.4f}")
        print(f"  Val   - Total: {avg_val_loss:.4f}, Forecast: {avg_val_forecast:.4f}, Recon: {avg_val_recon:.4f}")
        print("-" * 70)

        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
            print(f"Saved best model (val_loss: {avg_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth')


def load_model(checkpoint_path, Np=12, Nf=12, device='cuda'):
    """
    Load a trained DroneStalkerModel from checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        Np: Observation window size
        Nf: Prediction horizon
        device: 'cuda' or 'cpu'

    Returns:
        Loaded model in eval mode
    """
    model = DroneStalkerModel(Np=Np, Nf=Nf, input_dim=132, num_layers=4,
                             num_heads=4, dim_feedforward=512, dropout=0.1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model


def run_inference(model, observation_frames, mean_velocity, std_velocity, mean_coordinates, std_coordinates, device='cuda'):
    """
    Run inference on a sequence of observation frames.

    Args:
        model: Trained DroneStalkerModel
        observation_frames: List of Np consecutive frame dictionaries from FRED dataset
        mean_velocity, std_velocity, mean_coordinates, std_coordinates: Normalization stats from training
        device: 'cuda' or 'cpu'

    Returns:
        predicted_boxes: numpy array of shape [Nf, 4] containing predicted bounding boxes [x1, y1, x2, y2]
    """
    model.eval()
    Np = len(observation_frames)

    # === Process Event Images ===
    event_images = []
    for sample in observation_frames:
        event_tensor = get_event_feature(sample)
        if event_tensor is None:
            event_tensor = torch.zeros(64, 64)
        event_images.append(event_tensor.unsqueeze(0))
    event_images = torch.stack(event_images).unsqueeze(0)  # [1, Np, 1, 64, 64]

    # === Process Kinematic Features ===
    kinematic_features = []
    for i in range(len(observation_frames) - 1):
        features = get_kinematic_features(
            observation_frames[i], observation_frames[i + 1],
            mean_velocity, std_velocity, mean_coordinates, std_coordinates
        )
        kinematic_features.append(features)
    kinematic_features.append(kinematic_features[-1])  # Duplicate last velocity
    kinematic_features = torch.tensor([kinematic_features], dtype=torch.float32)  # [1, Np, 4]

    # Move to device
    event_images = event_images.to(device)
    kinematic_features = kinematic_features.to(device)

    # Run inference
    with torch.no_grad():
        reconstructed_past, predicted_future = model(event_images, kinematic_features)

    # Convert to numpy
    predicted_boxes = predicted_future.cpu().numpy()[0]  # [Nf, 4]

    return predicted_boxes


def visualize_predictions(observation_frames, predicted_boxes, save_path=None):
    """Visualize predicted trajectory."""
    import matplotlib.pyplot as plt

    # Observed trajectory
    obs_centers = [((f['x1'] + f['x2']) / 2, (f['y1'] + f['y2']) / 2) for f in observation_frames]
    obs_x, obs_y = zip(*obs_centers)

    # Predicted trajectory
    pred_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in predicted_boxes]
    pred_x, pred_y = zip(*pred_centers)

    plt.figure(figsize=(10, 8))
    plt.plot(obs_x, obs_y, 'bo-', label='Observed', linewidth=2, markersize=8)
    plt.plot(pred_x, pred_y, 'ro-', label='Predicted', linewidth=2, markersize=8)
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Drone Trajectory Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def inference_example():
    """Example of running inference with a trained model."""
    checkpoint_path = 'best_model.pth'
    Np, Nf = 12, 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("DroneStalker Inference Example")
    print("=" * 70)

    # Load model
    model = load_model(checkpoint_path, Np=Np, Nf=Nf, device=device)

    # Load data and compute normalization stats
    print("\nLoading dataset...")
    data = get_data()
    mean_velocity = get_mean_velocity(data)
    std_velocity = get_std_velocity(data, mean_velocity)
    mean_coordinates = get_mean_coordinates(data)
    std_coordinates = get_std_coordinates(data, mean_coordinates)

    # Select random sequence
    import random
    start_idx = random.randint(0, len(data) - Np - Nf)
    observation_frames = data[start_idx : start_idx + Np]
    ground_truth_future = data[start_idx + Np : start_idx + Np + Nf]

    print(f"\nRunning inference on frames {start_idx} to {start_idx + Np - 1}")

    # Run inference
    predicted_boxes = run_inference(model, observation_frames, mean_velocity, std_velocity,
                                   mean_coordinates, std_coordinates, device)

    print("\nPredictions:")
    print("Frame | Predicted [x1, y1, x2, y2] | Ground Truth [x1, y1, x2, y2]")
    print("-" * 70)
    for i in range(Nf):
        pred = predicted_boxes[i]
        gt = ground_truth_future[i]
        print(f"{i+1:5d} | [{pred[0]:6.1f}, {pred[1]:6.1f}, {pred[2]:6.1f}, {pred[3]:6.1f}] | "
              f"[{gt['x1']:6.1f}, {gt['y1']:6.1f}, {gt['x2']:6.1f}, {gt['y2']:6.1f}]")

    # Visualize
    visualize_predictions(observation_frames, predicted_boxes, save_path='prediction_viz.png')

    print("\nInference completed!")


def main():
    
    """Main training script."""
    # Configuration
    Np = 12  # Observation window
    Nf = 12  # Prediction horizon
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("DroneStalker Training - FRED Dataset")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Observation window (Np): {Np} frames")
    print(f"Prediction horizon (Nf): {Nf} frames")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print("=" * 70)

    # Load data
    print("\nLoading FRED dataset...")
    data = get_data()

    # Train/Val split (80/20)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    print(f"Train: {len(train_data)} frames, Val: {len(val_data)} frames")

    # Create datasets
    train_dataset = DroneTrajectoryDataset(train_data, Np=Np, Nf=Nf)
    val_dataset = DroneTrajectoryDataset(val_data, Np=Np, Nf=Nf)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device == 'cuda')
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device == 'cuda')
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing DroneStalkerModel...")
    model = DroneStalkerModel(Np=Np, Nf=Nf, input_dim=132, num_layers=4,
                             num_heads=4, dim_feedforward=512, dropout=0.1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


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