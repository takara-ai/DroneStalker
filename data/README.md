# FRED Dataset - DroneStalker Data Directory

This directory contains data from the **FRED (Florence RGB-Event Drone Dataset)**, a multimodal dataset for drone detection, tracking, and trajectory forecasting with spatiotemporally synchronized RGB and event camera data.

**Source**: [FRED Dataset Repository](https://github.com/miccunifi/FRED) | [Official Website](https://miccunifi.github.io/FRED/)

## Dataset Overview

FRED is a large-scale multimodal dataset featuring:

- **7+ hours** of densely annotated drone trajectories
- **5 different drone models** (e.g., DJI Mini 2, DarwinFPV cineape20)
- **Spatiotemporally synchronized** RGB and event camera streams
- **Challenging scenarios** including rain, adverse lighting, night conditions, and indoor environments
- **700,000+ annotated frames** per modality

## Directory Structure

Each sequence folder (e.g., `230/`) contains the following structure:

```
230/
├── RGB/                    # Original RGB video frames (JPG format)
│   └── Video_230_*.jpg     # 3,692 frames per sequence
│
├── PADDED_RGB/             # RGB frames with padding for coordinate alignment
│   └── Video_230_*.jpg     # Same frame count as RGB/ (3,692 frames)
│
├── Event/                  # Event camera data
│   ├── Frames/             # Event frames (PNG format)
│   │   └── Video_230_*.png # 3,759 frames per sequence
│   ├── events.raw          # Raw event stream data (binary format)
│   ├── events.raw.tmp_index # Index file for raw events
│   └── output_events.npz   # Processed event data in NumPy compressed format
│
├── RGB_YOLO/              # YOLO format annotations for RGB frames
│   └── Video_230_*.txt    # YOLO format: class x_center y_center width height (normalized)
│
├── Event_YOLO/            # YOLO format annotations for Event frames
│   └── Video_230_*.txt    # YOLO format: class x_center y_center width height (normalized)
│
├── Removed_frames/        # Frames removed during processing
│   └── Video_230_*.jpg    # ~207 removed RGB frames
│
├── coordinates.txt        # Bounding box annotations (extended boxes with padding)
│                          # Format: time: x1, y1, x2, y2
│                          # ~3,673 lines
│
├── interpolated_coordinates.txt  # Interpolated bounding boxes with IDs
│                                 # Format: time: x1, y1, x2, y2, id
│                                 # ~3,760 lines
│
└── tracks.txt             # Tracking data with detailed information
                           # Format: time, id, x, y, width, height
                           # ~3,119 lines
```

## Annotation Formats

### coordinates.txt

Bounding box annotations in extended coordinate space (includes padding):

```
time: x1, y1, x2, y2
```

**Example:**

```
0.0: 673.0, 315.0, 710.0, 340.0
0.033333: 673.0, 315.0, 710.0, 340.0
```

- **time**: Time relative to recording start in seconds.microseconds
- **x1, y1**: Top-left corner coordinates
- **x2, y2**: Bottom-right corner coordinates

### interpolated_coordinates.txt

Interpolated bounding boxes with identity tracking:

```
time: x1, y1, x2, y2, id
```

**Example:**

```
0.0: 673.0, 315.0, 710.0, 340.0, 1.0
0.033333: 673.0, 315.0, 710.0, 340.0, 1.0
```

- **id**: Unique identifier for the drone, consistent across frames

### tracks.txt

Detailed tracking information:

```
time, id, x, y, width, height
```

**Example:**

```
4.433289,19,686,308,21,21
4.466622,19,686,308,21,21
```

- **time**: Timestamp in seconds.microseconds
- **id**: Drone identifier
- **x, y**: Bounding box coordinates (likely top-left corner)
- **width, height**: Bounding box dimensions in pixels

## Data Modalities

### RGB Frames

- **Format**: JPG images
- **Naming**: `Video_{sequence_id}_{timestamp}.jpg`
- **Count**: 3,692 frames per sequence (30 FPS, ~2 minutes)
- **Purpose**: Standard RGB video for detection and tracking

### Event Frames

- **Format**: PNG images
- **Naming**: `Video_{sequence_id}_{timestamp}.png`
- **Count**: 3,759 frames per sequence (slightly higher temporal resolution than RGB)
- **Purpose**: Event camera frames with high temporal resolution and dynamic range

### Raw Event Data

- **events.raw**: Raw event stream from the event camera (binary data file)
- **output_events.npz**: Processed event data in NumPy compressed format (ZIP archive)
- **Purpose**: Access to raw event stream for custom processing

## Synchronization

The RGB and Event frames are **spatiotemporally synchronized**, meaning:

- RGB and Event frames can be perfectly overlapped
- Timestamps align between modalities
- Coordinate spaces are aligned (with padding in RGB frames)

## Use Cases

This dataset supports multiple computer vision tasks:

1. **Detection**: Object detection of drones in RGB and/or event streams
2. **Tracking**: Multi-object tracking with consistent identity across frames
3. **Trajectory Forecasting**: Predicting future drone trajectories
4. **Multimodal Learning**: Combining RGB and event data for improved performance

## Notes

- The `PADDED_RGB/` directory contains RGB frames with padding to enable coordinate space alignment with event frames
- The `coordinates.txt` file uses extended boxes (includes padding), while `coordinates_rgb.txt` (if present) excludes padding
- **YOLO-formatted directories** (`RGB_YOLO/` and `Event_YOLO/`) contain annotation files in YOLO format:
  - Each frame has a corresponding `.txt` file with the same name
  - Format: `class x_center y_center width height` (all values normalized 0-1)
  - Example: `0 0.540234375 0.4548611111111111 0.02890625 0.034722222222222224`
- **Removed_frames/** contains ~207 RGB frames that were removed during processing
- Frame counts vary between RGB (3,692) and Event (3,759) modalities due to different temporal resolutions
- Annotation file line counts: `coordinates.txt` (3,673), `interpolated_coordinates.txt` (3,760), `tracks.txt` (3,119)

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{magrini2025fred,
  title={FRED: The Florence RGB-Event Drone Dataset},
  author={Magrini, Gabriele and Marini, Niccol{`o} and Becattini, Federico and Berlincioni, Lorenzo and Biondi, Niccol{`o} and Pala, Pietro and Del Bimbo, Alberto},
  booktitle={Proceedings of the 33rd ACM International conference on multimedia},
  year={2025}
}
```

## License

This dataset is licensed under the Apache License 2.0.
