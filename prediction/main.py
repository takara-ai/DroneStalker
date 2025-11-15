from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN64x64(nn.Module):
    """
    Simple 3-layer CNN for 64x64 grayscale images.
    Takes input from get_event_feature and outputs a 128-dimensional feature vector.
    """
    def __init__(self):
        super(CNN64x64, self).__init__()

        # Layer 1: (1, 64, 64) -> (32, 32, 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: (32, 32, 32) -> (64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: (64, 16, 16) -> (128, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Output: Flatten to 128-dim vector
        self.fc = nn.Linear(128 * 8 * 8, 128)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, 64, 64)

        Returns:
            torch.Tensor: Output tensor of shape (batch, 128)
        """
        # Conv layer 1
        x = self.pool1(F.relu(self.conv1(x)))

        # Conv layer 2
        x = self.pool2(F.relu(self.conv2(x)))

        # Conv layer 3
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten and output 128-dim vector
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)

        return x


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
def get_center_coordinates(sample: dict) -> tuple(int, int):
    x1, y1, x2, y2 = sample["x1"], sample["y1"], sample["x2"], sample["y2"]
    xc = (x1 + x2) / 2 # Center x coordinate of bounding box
    yc = (y1 + y2) / 2 # Center y coordinate of bounding box
    return (xc, yc)
    
# Get the velocity for two samples
def get_kinematic_features(sample1: dict, sample2: dict, meanv: float, stdv: float, mean_coordinates: tuple(float, float), std_coordinates: tuple(float, float)) -> tuple(float, float):
    x1, y1 = get_center_coordinates(sample1)
    x2, y2 = get_center_coordinates(sample2)
    dx = x2 - x1
    dy = y2 - y1
    vx = dx / (sample2["time"] - sample1["time"]) # Velocity in pixels per second (px/s)
    vy = dy / (sample2["time"] - sample1["time"]) # Velocity in pixels per second (px/s)
    
    # Perform Z-score standardisation
    x_zscore = (x2 - mean_coordinates[0]) / std_coordinates[0]
    y_zscore = (y2 - mean_coordinates[1]) / std_coordinates[1]
    vx_zscore = (vx - meanv) / stdv
    vy_zscore = (vy - meanv) / stdv
    return (x_zscore, y_zscore, vx_zscore, vy_zscore) # Return the x2, y2, vx and vy (z-score standardised)

def get_mean_velocity(data: list[dict]) -> float:
    total_vx = 0
    total_vy = 0
    for i in range(len(data) - 1):
        vx, vy = get_kinematic_features(data[i], data[i + 1])
        total_vx += vx
        total_vy += vy
    mean_vx = total_vx / (len(data) - 1)
    mean_vy = total_vy / (len(data) - 1)
    return (mean_vx, mean_vy)

def get_std_velocity(data: list[dict], mean: float) -> float:
    return sqrt((sum((vx - mean)**2 for vx, vy in get_kinematic_features(data[i], data[i + 1]) for i in range(len(data) - 1)) / (len(data) - 1)) / (len(data) - 2))

def get_mean_coordinates(data: list[dict]) -> tuple(float, float):
    total_x = 0
    total_y = 0
    for i in range(len(data)):
        x, y = get_center_coordinates(data[i])
        total_x += x
        total_y += y
    mean_x = total_x / len(data)
    mean_y = total_y / len(data)
    return (mean_x, mean_y)

def get_std_coordinates(data: list[dict], mean: tuple(float, float)) -> tuple(float, float):
    std_x = sqrt((sum((x - mean[0])**2 for x, y in get_center_coordinates(data[i]) for i in range(len(data))) / len(data)) / (len(data) - 1))
    std_y = sqrt((sum((y - mean[1])**2 for x, y in get_center_coordinates(data[i]) for i in range(len(data))) / len(data)) / (len(data) - 1))
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
        resized_img = cropped_img.resize((64, 64), Image.ANTIALIAS)
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