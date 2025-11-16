#!/usr/bin/env python3
"""
Generate clean overlay frames for video visualization of trajectory predictions.

This script:
- Runs inference on every frame (rolling window)
- Generates clean white-background frames with colored trajectory arrows
- No labels, legends, or borders
- High frame rate output
- Progress bar with tqdm
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Import from main.py
from main import (
    get_data,
    load_model,
    run_inference,
    get_mean_velocity,
    get_std_velocity,
    get_mean_coordinates,
    get_std_coordinates,
)


def draw_arrow(canvas, pt1, pt2, color, thickness=2, arrow_scale=0.3):
    """
    Draw an arrow from pt1 to pt2 on canvas.
    
    Args:
        canvas: Image array to draw on
        pt1: Start point (x, y)
        pt2: End point (x, y)
        color: BGR color tuple
        thickness: Line thickness
        arrow_scale: Scale of arrow head relative to line length
    """
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    
    # Draw the line
    cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    # Calculate arrow head
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        # Normalize direction
        dx /= length
        dy /= length
        
        # Arrow head size
        arrow_length = min(length * arrow_scale, 15)
        arrow_angle = 25 * np.pi / 180  # 25 degrees
        
        # Calculate arrow head points
        angle1 = np.arctan2(dy, dx) + np.pi - arrow_angle
        angle2 = np.arctan2(dy, dx) + np.pi + arrow_angle
        
        p1 = (
            int(pt2[0] + arrow_length * np.cos(angle1)),
            int(pt2[1] + arrow_length * np.sin(angle1))
        )
        p2 = (
            int(pt2[0] + arrow_length * np.cos(angle2)),
            int(pt2[1] + arrow_length * np.sin(angle2))
        )
        
        # Draw arrow head
        cv2.line(canvas, pt2, p1, color, thickness, cv2.LINE_AA)
        cv2.line(canvas, pt2, p2, color, thickness, cv2.LINE_AA)


def draw_path_with_arrows(canvas, points, color, thickness=2, dot_radius=3):
    """
    Draw a path with dots and arrows indicating direction.
    
    Args:
        canvas: Image array to draw on
        points: Array of (x, y) points
        color: BGR color tuple
        thickness: Line thickness
        dot_radius: Radius of dots at each point
    """
    if len(points) < 1:
        return
    
    # Draw dots at each point
    for pt in points:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), dot_radius, color, -1, cv2.LINE_AA)
    
    # Draw arrows between consecutive points
    if len(points) >= 2:
        for i in range(len(points) - 1):
            draw_arrow(canvas, points[i], points[i + 1], color, thickness, arrow_scale=0.25)


def generate_frames(
    checkpoint_path='best_model.pth',
    out_dir='outputs/video_frames',
    Np=12,
    Nf=12,
    device='cuda',
    trail_length=24
):
    """
    Generate video frames with trajectory predictions.
    
    Args:
        checkpoint_path: Path to model checkpoint
        out_dir: Output directory for frames
        Np: Observation window size
        Nf: Prediction horizon
        device: 'cuda' or 'cpu'
        trail_length: Number of recent observed frames to show (default: 24)
    """
    print("=" * 70)
    print("Generating Video Frames for Trajectory Predictions")
    print("=" * 70)
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    print(f"Observation window (Np): {Np}")
    print(f"Prediction horizon (Nf): {Nf}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, Np=Np, Nf=Nf, device=device)
    print()
    
    # Load data
    print("Loading dataset...")
    data = get_data()
    print(f"Loaded {len(data)} frames")
    print()
    
    # Compute normalization statistics
    print("Computing normalization statistics...")
    mean_velocity = get_mean_velocity(data)
    std_velocity = get_std_velocity(data, mean_velocity)
    mean_coordinates = get_mean_coordinates(data)
    std_coordinates = get_std_coordinates(data, mean_coordinates)
    stats = (mean_velocity, std_velocity, mean_coordinates, std_coordinates)
    print()
    
    # Determine canvas size from first available image
    print("Determining canvas size...")
    ref_img = None
    for f in data:
        img_path = f.get('img_path')
        if img_path and os.path.exists(img_path):
            ref_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if ref_img is not None:
                break
    
    if ref_img is None:
        h, w = 1080, 1920
        print(f"Warning: No images found. Using default {w}x{h} canvas.")
    else:
        h, w = ref_img.shape[:2]
        print(f"Canvas size: {w}x{h}")
    print()
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Precompute all observed centers for efficiency
    obs_all = np.array([
        ((f['x1'] + f['x2']) / 2.0, (f['y1'] + f['y2']) / 2.0)
        for f in data
    ], dtype=np.float32)
    
    # Colors (BGR): royalblue=observed, crimson=predicted, limegreen=ground truth
    col_observed = (225, 105, 65)    # royalblue
    col_predicted = (60, 20, 220)    # crimson
    col_gt = (50, 205, 50)           # limegreen
    thickness = 2
    dot_radius = 3
    
    # Generate frames for each timestep
    print("Generating frames...")
    print(f"Total frames to generate: {len(data)}")
    print()
    
    frames_written = 0
    
    # Use tqdm for progress bar
    for t in tqdm(range(len(data)), desc="Processing frames", unit="frame"):
        # Create white background
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        
        # Collect observed trajectory (only recent trail_length frames)
        start_trail = max(0, t + 1 - trail_length)
        obs_trajectory = obs_all[start_trail:t+1]
        
        # Run prediction if we have enough observation frames
        if t >= Np:
            # Use last Np frames as observation window
            start_idx = t - Np + 1
            obs_frames = data[start_idx:t+1]
            
            # Run inference
            with torch.no_grad():
                pred_centers = run_inference(
                    model,
                    obs_frames,
                    mean_velocity,
                    std_velocity,
                    mean_coordinates,
                    std_coordinates,
                    device=device
                )
            
            # Get ground truth future if available
            gt_future = []
            for i in range(1, min(Nf + 1, len(data) - t)):
                gt_idx = t + i
                if gt_idx < len(data):
                    gt_future.append(obs_all[gt_idx])
            gt_future = np.array(gt_future, dtype=np.float32) if gt_future else None
            
            # Prepend last observed point to predicted trajectory for smooth connection
            if pred_centers is not None and len(pred_centers) > 0:
                last_obs_pt = obs_all[t:t+1]  # Current position
                pred_with_start = np.vstack([last_obs_pt, pred_centers])
            else:
                pred_with_start = None
            
            # Draw ground truth future (if available)
            if gt_future is not None and len(gt_future) > 0:
                # Also prepend last observed point to ground truth
                gt_with_start = np.vstack([obs_all[t:t+1], gt_future])
                draw_path_with_arrows(canvas, gt_with_start, col_gt, thickness, dot_radius)
            
            # Draw predicted trajectory
            if pred_with_start is not None and len(pred_with_start) > 0:
                draw_path_with_arrows(canvas, pred_with_start, col_predicted, thickness, dot_radius)
        
        # Draw observed trajectory (always drawn, up to current frame)
        if len(obs_trajectory) > 0:
            draw_path_with_arrows(canvas, obs_trajectory, col_observed, thickness, dot_radius)
        
        # Save frame with timestep in filename
        frame_filename = f"frame_{t:06d}.png"
        out_path = os.path.join(out_dir, frame_filename)
        cv2.imwrite(out_path, canvas)
        frames_written += 1
    
    print()
    print("=" * 70)
    print(f"✓ Successfully generated {frames_written} frames")
    print(f"Output directory: {out_dir}")
    print("=" * 70)
    print()
    print("To create a video, run:")
    print(f"  ffmpeg -framerate 30 -i {out_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate video frames for trajectory prediction visualization'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_model.pth',
        help='Path to model checkpoint (default: best_model.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/video_frames',
        help='Output directory for frames (default: outputs/video_frames)'
    )
    parser.add_argument(
        '--Np',
        type=int,
        default=12,
        help='Observation window size (default: 12)'
    )
    parser.add_argument(
        '--Nf',
        type=int,
        default=12,
        help='Prediction horizon (default: 12)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--trail',
        type=int,
        default=24,
        help='Length of observed trajectory trail in frames (default: 24)'
    )
    
    args = parser.parse_args()
    
    generate_frames(
        checkpoint_path=args.checkpoint,
        out_dir=args.output,
        Np=args.Np,
        Nf=args.Nf,
        device=args.device,
        trail_length=args.trail
    )


if __name__ == "__main__":
    main()
