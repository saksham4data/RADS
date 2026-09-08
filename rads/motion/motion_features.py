import numpy as np
from typing import Dict, List, Any
from rads.motion.trajectory import TrackHistory

def compute_motion_features(track_history: TrackHistory) -> Dict[int, Dict[str, Any]]:
    """
    Computes per-object motion features over its trajectory.
    Applies light smoothing to reduce jitter.
    Everything is in image-space (pixels).
    """
    features = {}
    
    for t_id in track_history.get_all_track_ids():
        traj = track_history.get_trajectory(t_id)
        if len(traj) < 2:
            # Not enough data for motion
            features[t_id] = {
                "total_displacement": 0.0,
                "net_displacement": 0.0,
                "mean_velocity": 0.0,
                "max_velocity": 0.0,
                "mean_acceleration": 0.0,
                "max_acceleration": 0.0,
                "mean_direction": [0.0, 0.0]
            }
            continue
            
        # Extract numpy arrays for vectorized operations
        cx = np.array([pt['cx'] for pt in traj])
        cy = np.array([pt['cy'] for pt in traj])
        timestamps = np.array([pt['timestamp'] for pt in traj])
        
        # We need distinct timestamps to avoid division by zero. If fps is perfectly constant, dt is constant.
        dt = np.diff(timestamps)
        # Avoid division by zero in case of identical timestamps
        dt[dt <= 0.001] = 0.001
        
        # Raw positional differences
        dx = np.diff(cx)
        dy = np.diff(cy)
        
        # Displacement
        step_displacement = np.sqrt(dx**2 + dy**2)
        total_displacement = float(np.sum(step_displacement))
        
        net_dx = cx[-1] - cx[0]
        net_dy = cy[-1] - cy[0]
        net_displacement = float(np.sqrt(net_dx**2 + net_dy**2))
        
        # Velocity (pixels / second)
        vx = dx / dt
        vy = dy / dt
        
        # Light smoothing on velocity (moving average window of 3)
        window = min(3, len(vx))
        if window > 1:
            vx = np.convolve(vx, np.ones(window)/window, mode='same')
            vy = np.convolve(vy, np.ones(window)/window, mode='same')
            
        speed = np.sqrt(vx**2 + vy**2)
        mean_velocity = float(np.mean(speed))
        max_velocity = float(np.max(speed))
        
        # Acceleration (pixels / second^2)
        if len(vx) >= 2:
            ax = np.diff(vx) / dt[1:]
            ay = np.diff(vy) / dt[1:]
            
            # Light smoothing on acceleration
            window_acc = min(3, len(ax))
            if window_acc > 1:
                ax = np.convolve(ax, np.ones(window_acc)/window_acc, mode='same')
                ay = np.convolve(ay, np.ones(window_acc)/window_acc, mode='same')
                
            acceleration_mag = np.sqrt(ax**2 + ay**2)
            mean_accel = float(np.mean(acceleration_mag))
            max_accel = float(np.max(acceleration_mag))
        else:
            mean_accel = 0.0
            max_accel = 0.0
            
        # Direction
        if net_displacement > 0:
            dir_x = float(net_dx / net_displacement)
            dir_y = float(net_dy / net_displacement)
        else:
            dir_x, dir_y = 0.0, 0.0
            
        features[t_id] = {
            "total_displacement": total_displacement,
            "net_displacement": net_displacement,
            "mean_velocity": mean_velocity,
            "max_velocity": max_velocity,
            "mean_acceleration": mean_accel,
            "max_acceleration": max_accel,
            "mean_direction": [dir_x, dir_y]
        }
        
    return features
