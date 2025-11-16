import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GRUTrajectoryModel(nn.Module):
    """
    GRU-based model predicting midpoint (center) trajectories.

    Inputs:
      - kinematic_features: [batch, Np, 4] where 4 = [x_z, y_z, vx_z, vy_z]

    Outputs:
      - reconstructed_past: [batch, Np, 2] (center x_z, y_z)
      - predicted_future:   [batch, Nf, 2] (center x_z, y_z)
    """
    def __init__(self, Np=12, Nf=12, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.Np = Np
        self.Nf = Nf
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder processes kinematic features over the observation window
        self.encoder = nn.GRU(
            input_size=4, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        # Reconstruction head maps encoder outputs to center (x_z, y_z)
        self.past_head = nn.Linear(hidden_size, 2)

        # Decoder autoregressively predicts future centers
        self.decoder_cell = nn.GRUCell(input_size=2, hidden_size=hidden_size)
        self.future_head = nn.Linear(hidden_size, 2)

    def forward(self, kinematic_features):
        batch_size, Np, _ = kinematic_features.shape

        # Encode observation window
        enc_out, h_n = self.encoder(kinematic_features)  # enc_out: [B, Np, H], h_n: [L, B, H]
        # Reconstruct past centers from encoder outputs
        reconstructed_past = self.past_head(enc_out)  # [B, Np, 2]

        # Initialize decoder hidden state from last encoder layer's hidden
        h = h_n[-1]  # [B, H]
        # Start token: use last reconstructed center as initial input
        y_prev = reconstructed_past[:, -1, :]  # [B, 2]

        future_outputs = []
        for _ in range(self.Nf):
            h = self.decoder_cell(y_prev, h)        # [B, H]
            y = self.future_head(h)                 # [B, 2]
            future_outputs.append(y)
            y_prev = y

        predicted_future = torch.stack(future_outputs, dim=1)  # [B, Nf, 2]

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
        # L_Nf: Forecasting loss (RMSE over centers)
        forecasting_loss = torch.sqrt(self.mse_loss(predicted_future, ground_truth_future))

        # L_Np: Reconstruction loss (RMSE over centers)
        reconstruction_loss = torch.sqrt(self.mse_loss(reconstructed_past, ground_truth_past))

        # Total loss: L = L_Nf + λ * L_Np
        total_loss = forecasting_loss + self.lambda_recon * reconstruction_loss

        return total_loss, forecasting_loss, reconstruction_loss


def get_data() -> list[dict]:
    # Load dataset
    raw_data = []
    for x in [0, 1]:
        with open(f"../data/{x}/coordinates.txt", "r") as coord_file:
            lines = coord_file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Expect format: "<time>: x1, y1, x2, y2, ...optional fields"
            try:
                time_part, coords_part = line.split(":", 1)
            except ValueError:
                continue
            time_str = time_part.strip()

            # Only take the first 4 comma-separated values as bbox coords
            coord_tokens = [c.strip() for c in coords_part.split(",")]
            if len(coord_tokens) < 4:
                continue
            coord_values = coord_tokens[:4]

            try:
                time_val = float(time_str)
                x1, y1, x2, y2 = [float(c) for c in coord_values]
            except ValueError:
                continue

            # Frame filenames are built by removing the decimal point from the timestamp
            frame_time_str = time_str.replace(".", "")
            img_path = f"../data/{x}/Event/Frames/Video_{x}_frame_{frame_time_str}.png"

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

# Images are no longer used; model operates purely on coordinates.

## Positional encoding is not required for the GRU-only pipeline.


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

        # === Ground Truth Centers (z-scored) ===
        def z_center(sample):
            x, y = get_center_coordinates(sample)
            xz = (x - self.mean_coordinates[0]) / self.std_coordinates[0]
            yz = (y - self.mean_coordinates[1]) / self.std_coordinates[1]
            return [xz, yz]

        ground_truth_past = torch.tensor([z_center(s) for s in obs_frames], dtype=torch.float32)     # [Np, 2]
        ground_truth_future = torch.tensor([z_center(s) for s in future_frames], dtype=torch.float32) # [Nf, 2]

        return kinematic_features, ground_truth_past, ground_truth_future


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda'):
    """
    Training loop for GRUTrajectoryModel with FRED loss (forecasting + reconstruction).
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

        for batch_idx, (kinematic_features, gt_past, gt_future) in enumerate(train_loader):
            kinematic_features = kinematic_features.to(device)
            gt_past = gt_past.to(device)
            gt_future = gt_future.to(device)

            # Forward pass
            reconstructed_past, predicted_future = model(kinematic_features)

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
            for kinematic_features, gt_past, gt_future in val_loader:
                kinematic_features = kinematic_features.to(device)
                gt_past = gt_past.to(device)
                gt_future = gt_future.to(device)

                reconstructed_past, predicted_future = model(kinematic_features)
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
    Load a trained GRUTrajectoryModel from checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        Np: Observation window size
        Nf: Prediction horizon
        device: 'cuda' or 'cpu'

    Returns:
        Loaded model in eval mode
    """
    model = GRUTrajectoryModel(Np=Np, Nf=Nf, hidden_size=128, num_layers=2, dropout=0.1)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Warning: failed to load checkpoint '{checkpoint_path}' for GRU model: {e}")
        print("Proceeding with randomly initialized weights.")
        checkpoint = {}
    model = model.to(device)
    model.eval()

    if checkpoint:
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
        model: Trained GRUTrajectoryModel
        observation_frames: List of Np consecutive frame dictionaries from FRED dataset
        mean_velocity, std_velocity, mean_coordinates, std_coordinates: Normalization stats from training
        device: 'cuda' or 'cpu'

    Returns:
        predicted_centers: numpy array of shape [Nf, 2] containing predicted midpoint coordinates [x, y] in pixels
    """
    model.eval()
    Np = len(observation_frames)

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
    kinematic_features = kinematic_features.to(device)

    # Run inference
    with torch.no_grad():
        _, predicted_future_z = model(kinematic_features)  # [1, Nf, 2] (z-scored)

    # Un-normalize centers
    pred_z = predicted_future_z.cpu().numpy()[0]  # [Nf, 2]
    pred_x = pred_z[:, 0] * std_coordinates[0] + mean_coordinates[0]
    pred_y = pred_z[:, 1] * std_coordinates[1] + mean_coordinates[1]
    predicted_centers = np.stack([pred_x, pred_y], axis=1)

    return predicted_centers


def visualize_predictions(observation_frames, predicted_centers, ground_truth_future=None, save_path=None):
    """Visualize predicted trajectory."""
    import matplotlib.pyplot as plt

    # Observed trajectory
    obs_centers = [((f['x1'] + f['x2']) / 2, (f['y1'] + f['y2']) / 2) for f in observation_frames]
    obs_x, obs_y = zip(*obs_centers)

    # Predicted trajectory (already centers)
    pred_x, pred_y = zip(*predicted_centers)
    # Ground truth future centers if provided (list[dict] or array-like)
    gt_x = gt_y = None
    if ground_truth_future is not None:
        if isinstance(ground_truth_future, (list, tuple)) and len(ground_truth_future) > 0 and isinstance(ground_truth_future[0], dict):
            gt_centers = [((g['x1'] + g['x2']) / 2, (g['y1'] + g['y2']) / 2) for g in ground_truth_future]
        else:
            gt_centers = list(ground_truth_future)
        gt_x, gt_y = zip(*gt_centers)

    plt.figure(figsize=(10, 8))
    plt.plot(obs_x, obs_y, 'bo-', label='Observed', linewidth=2, markersize=8)
    plt.plot(pred_x, pred_y, 'ro-', label='Predicted', linewidth=2, markersize=8)
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Drone Trajectory Prediction')
    if gt_x is not None:
        plt.plot(gt_x, gt_y, 'go-', label='Ground Truth', linewidth=2, markersize=8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_overlay_on_image(observation_frames, predicted_centers, save_path, ground_truth_future=None):
    """
    Overlay observed and predicted center trajectories on the last observed image.

    Args:
        observation_frames: list of dicts with bbox coords and 'img_path'
        predicted_centers: ndarray/List [Nf, 2] of pixel centers (x, y)
        save_path: output image path
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import matplotlib.patches as patches

    if not observation_frames:
        print("No observation frames provided for overlay.")
        return

    # Use the last observed frame as the backdrop
    bg_img_path = observation_frames[-1].get('img_path')
    try:
        bg = Image.open(bg_img_path)
    except Exception as e:
        print(f"Failed to open background image {bg_img_path}: {e}")
        return

    # Observed centers (in pixels)
    obs_centers = np.array([((f['x1'] + f['x2']) / 2.0, (f['y1'] + f['y2']) / 2.0) for f in observation_frames])
    pred_centers = np.array(predicted_centers)
    gt_centers = None
    if ground_truth_future is not None:
        if isinstance(ground_truth_future, (list, tuple)) and len(ground_truth_future) > 0 and isinstance(ground_truth_future[0], dict):
            gt_centers = np.array([((g['x1'] + g['x2']) / 2.0, (g['y1'] + g['y2']) / 2.0) for g in ground_truth_future])
        else:
            gt_centers = np.array(ground_truth_future)

    plt.figure(figsize=(10, 8))
    plt.imshow(bg)
    ax = plt.gca()

    # Highlight the drone in the background by drawing the last observed bbox
    last = observation_frames[-1]
    bbox_w = last['x2'] - last['x1']
    bbox_h = last['y2'] - last['y1']
    rect = patches.Rectangle((last['x1'], last['y1']), bbox_w, bbox_h,
                             linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.9, label='Last bbox')
    ax.add_patch(rect)

    # Plot observed path with markers
    plt.plot(obs_centers[:, 0], obs_centers[:, 1], 'o-', color='royalblue', label='Observed', linewidth=2, markersize=5, alpha=0.95)
    # Plot predicted path with markers
    plt.plot(pred_centers[:, 0], pred_centers[:, 1], 'o-', color='crimson', label='Predicted', linewidth=2, markersize=5, alpha=0.95)
    # Plot ground truth future if available
    if gt_centers is not None and len(gt_centers) > 0:
        plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'o-', color='limegreen', label='Ground Truth', linewidth=2, markersize=5, alpha=0.95)

    # Add arrows along the paths to indicate direction
    def add_arrows(points, color):
        if len(points) < 2:
            return
        xs = points[:-1, 0]
        ys = points[:-1, 1]
        dx = points[1:, 0] - points[:-1, 0]
        dy = points[1:, 1] - points[:-1, 1]
        plt.quiver(xs, ys, dx, dy, angles='xy', scale_units='xy', scale=1.0,
                   color=color, width=0.004, alpha=0.9, headwidth=6, headlength=8, headaxislength=7)

    add_arrows(obs_centers, 'royalblue')
    add_arrows(pred_centers, 'crimson')
    if gt_centers is not None and len(gt_centers) > 1:
        add_arrows(gt_centers, 'limegreen')
    plt.title('Trajectory Overlay')
    plt.legend()
    plt.axis('off')
    # Invert y to match image coordinate system if axes are not image-based
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def render_overlay_frames_opencv(observation_frames, predicted_centers, out_dir, ground_truth_future=None):
    """
    Render a sequence of PNG frames on a white background using OpenCV.

    - Canvas size is taken from the last observed frame's image size.
    - Draws observed path (full) and predicted path incrementally per frame.
    - Uses a simple monochrome style suitable for later overlay.

    Args:
        observation_frames: list[dict] with bbox coords and 'img_path'
        predicted_centers: array-like [Nf, 2] pixel centers
        out_dir: directory to write frames into
        ground_truth_future: optional list[dict] future frames (unused here)
    """
    import os
    import cv2
    import numpy as np

    if not observation_frames:
        print("No observation frames provided for rendering.")
        return

    # Determine canvas size from the last observation image
    bg_path = observation_frames[-1].get('img_path')
    img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback to a default canvas if image is missing
        h, w = 1080, 1920
        print(f"Warning: failed to read '{bg_path}'. Using default canvas {w}x{h}.")
    else:
        h, w = img.shape[:2]

    os.makedirs(out_dir, exist_ok=True)

    # Prepare points
    obs_pts = np.array([[(f['x1'] + f['x2']) / 2.0, (f['y1'] + f['y2']) / 2.0] for f in observation_frames], dtype=np.float32)
    pred_pts = np.array(predicted_centers, dtype=np.float32)

    # Styles (monochrome)
    col = (0, 0, 0)  # black
    obs_col = (64, 64, 64)  # dark gray for observed
    thick = 2
    radius = 3

    # Draw Nf frames, where frame k shows predicted path up to k
    for k in range(len(pred_pts)):
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)  # white background

        # Observed path
        if len(obs_pts) >= 2:
            cv2.polylines(canvas, [obs_pts.astype(np.int32)], isClosed=False, color=obs_col, thickness=thick)
        for p in obs_pts:
            cv2.circle(canvas, (int(p[0]), int(p[1])), radius, obs_col, -1)

        # Predicted path up to k
        seg = pred_pts[: k + 1]
        if len(seg) >= 2:
            cv2.polylines(canvas, [seg.astype(np.int32)], isClosed=False, color=col, thickness=thick)
        # Current predicted point
        cp = seg[-1]
        cv2.circle(canvas, (int(cp[0]), int(cp[1])), radius + 1, col, -1)

        out_path = os.path.join(out_dir, f"overlay_{k:06d}.png")
        cv2.imwrite(out_path, canvas)
    print(f"Saved {len(pred_pts)} overlay frames to {out_dir}")


def render_realtime_sequence_opencv(data, model, stats, out_dir, Np=12, Nf=12, device='cuda', decay=0.90):
    """
    Render per-frame white-background overlays across the dataset using a rolling
    window: for each time t, use the last Np observations to forecast the next Nf
    centers, and draw the full observed path up to t plus the future path.

    Args:
        data: list[dict] of frames (from get_data())
        model: loaded GRUTrajectoryModel
        stats: tuple(mean_v, std_v, mean_c, std_c)
        out_dir: output directory for PNG frames
        Np, Nf: observation and forecast horizons
        device: 'cuda' or 'cpu'
    """
    import os
    import cv2
    import numpy as np

    mean_v, std_v, mean_c, std_c = stats
    os.makedirs(out_dir, exist_ok=True)

    if len(data) < Np + 1:
        print("Not enough data for streaming overlays.")
        return

    # Determine canvas size from the first available image
    ref_img = None
    for f in data:
        p = f.get('img_path')
        if p:
            ref_img = cv2.imread(p, cv2.IMREAD_COLOR)
            if ref_img is not None:
                break
    if ref_img is None:
        h, w = 1080, 1920
        print("Warning: No images found. Using default 1920x1080 canvas.")
    else:
        h, w = ref_img.shape[:2]

    # Precompute observed centers for speed
    obs_all = np.array([((f['x1'] + f['x2']) / 2.0, (f['y1'] + f['y2']) / 2.0) for f in data], dtype=np.float32)

    # Colors (BGR): blue=context(observed), red=predicted, green=ground truth
    col_ctx = (255, 0, 0)   # blue
    col_pred = (0, 0, 255)  # red
    col_gt = (0, 255, 0)    # green
    thick = 2
    radius = 2

    frames_written = 0
    # Accumulated fading trail (float in [0,1], 0=no ink, 1=full ink); we subtract from white
    trail = np.zeros((h, w, 3), dtype=np.float32)

    # Iterate t across all frames; from t>=Np we forecast, earlier we render observed only
    for t in range(0, len(data)):
        if t >= Np:
            start = t - Np
            obs_frames = data[start:t]
            pred_centers = run_inference(
                model,
                obs_frames,
                mean_v,
                std_v,
                mean_c,
                std_c,
                device=device,
            )
        else:
            pred_centers = []

        # Fade the existing trail
        trail *= float(decay)

        # Prepare a fresh draw layer (float 0..1)
        draw = np.zeros_like(trail, dtype=np.float32)

        # Draw context: just the newest observed segment (t-1 -> t)
        if t >= 1:
            p0 = obs_all[t - 1]
            p1 = obs_all[t]
            cv2.line(draw, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     color=(col_ctx[0]/255.0, col_ctx[1]/255.0, col_ctx[2]/255.0), thickness=thick)
            cv2.circle(draw, (int(p1[0]), int(p1[1])), radius, (col_ctx[0]/255.0, col_ctx[1]/255.0, col_ctx[2]/255.0), -1)

        # Draw predicted path for current t
        fut = np.array(pred_centers, dtype=np.float32)
        if len(fut) >= 2:
            pts = fut.astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(draw, (int(pts[i][0]), int(pts[i][1])), (int(pts[i+1][0]), int(pts[i+1][1])),
                         color=(col_pred[0]/255.0, col_pred[1]/255.0, col_pred[2]/255.0), thickness=thick)
            cv2.circle(draw, (int(pts[0][0]), int(pts[0][1])), radius+1, (col_pred[0]/255.0, col_pred[1]/255.0, col_pred[2]/255.0), -1)

        # Draw ground truth future if available fully within bounds
        if t + 1 < len(data):
            gt_end = min(len(data), t + Nf + 1)
            gt_seq = obs_all[t:gt_end]
            if len(gt_seq) >= 2:
                pts = gt_seq.astype(np.int32)
                for i in range(len(pts) - 1):
                    cv2.line(draw, (int(pts[i][0]), int(pts[i][1])), (int(pts[i+1][0]), int(pts[i+1][1])),
                             color=(col_gt[0]/255.0, col_gt[1]/255.0, col_gt[2]/255.0), thickness=thick)

        # Accumulate and clip
        trail = np.clip(trail + draw, 0.0, 1.0)

        # Composite over white background
        canvas = (255.0 - trail * 255.0).astype(np.uint8)

        # Name output to match current Event frame when available
        base_name = None
        cur_img_path = data[t].get('img_path')
        if cur_img_path:
            base_name = os.path.basename(cur_img_path)
        out_fname = base_name if base_name else f"overlay_rt_{t:06d}.png"
        out_path = os.path.join(out_dir, out_fname)
        cv2.imwrite(out_path, canvas)
        frames_written += 1

    print(f"Saved {frames_written} real-time overlay frames to {out_dir}")


def inference_example():
    """Example of running inference with a trained model."""
    checkpoint_path = 'checkpoint_epoch_30.pth'
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
    predicted_centers = run_inference(model, observation_frames, mean_velocity, std_velocity,
                                      mean_coordinates, std_coordinates, device)

    print("\nPredictions:")
    print("Frame | Predicted center [x, y] | GT center [x, y]")
    print("-" * 70)
    for i in range(Nf):
        predx, predy = predicted_centers[i]
        gt_x = (ground_truth_future[i]['x1'] + ground_truth_future[i]['x2']) / 2
        gt_y = (ground_truth_future[i]['y1'] + ground_truth_future[i]['y2']) / 2
        print(f"{i+1:5d} | [{predx:6.1f}, {predy:6.1f}] | "
              f"[{gt_x:6.1f}, {gt_y:6.1f}]")

    # Visualize
    visualize_predictions(observation_frames, predicted_centers, ground_truth_future, save_path='prediction_viz.png')
    visualize_overlay_on_image(observation_frames, predicted_centers, save_path='prediction_overlay.png', ground_truth_future=ground_truth_future)

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
    print("\nInitializing GRUTrajectoryModel (midpoint paths)...")
    model = GRUTrajectoryModel(Np=Np, Nf=Nf, hidden_size=128, num_layers=2, dropout=0.1)

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
    inference_example()

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
# Notes:
# This GRU-only pipeline uses only coordinate-derived features. No images/CNN/transformer.
'''
