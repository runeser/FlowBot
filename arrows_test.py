import cv2
import numpy as np
from datetime import datetime

from src.flow_handler import calc_points, draw_arrows
from src.logger import log
from src.save_handler import save_arrow_frame
from src.load_handler import load_vid1


# def rotate_180(frame):
    # return cv2.rotate(frame, cv2.ROTATE_180)

def visualise_flow_test(num_frames=150, frame_interval=30):
    # Load all frames
    frames = load_vid1()
    date_str = datetime.now()
    log.info(f"Loaded {len(frames)} frames total")
    
    if len(frames) < 2:
        log.warning(f"Failed to load frames, got {len(frames)} frames")
        return
    
    processed_frames = 0
    arrow_frames = []
    
    # Subsample: take every 'frame_interval' frames
    for i in range(0, len(frames) - frame_interval, frame_interval):
        prev_frame = frames[i]
        curr_frame = frames[i + frame_interval]
        
        # Rotate frames 180 degrees
        # prev_frame = rotate_180(prev_frame)
        # curr_frame = rotate_180(curr_frame)
        
        
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        p1, p2 = calc_points(gray_prev, gray_curr)
        
        if p1 is not None and len(p1) > 0:
            
            frame_with_arrows = draw_arrows(curr_frame, p1, p2)
            arrow_frames.append(frame_with_arrows)
            
            
            img_name = f"{date_str}_frame_{processed_frames:04d}.png"
            save_arrow_frame(img_name, frame_with_arrows)
            
            log.info(f"Frame {processed_frames + 1}: {len(p1)} corners tracked (skip={frame_interval})")
            processed_frames += 1
        else:
            log.warning(f"No points found for frame {i} -> {i + frame_interval}")
    
    if not arrow_frames:
        log.warning("No frames processed")
        return
    
    curr_idx = 0
    
    def log_idx():
        log.info(f"Frame {curr_idx + 1}/{len(arrow_frames)}")
    
    
    # Create window
    cv2.namedWindow('Optical Flow Visualization', cv2.WINDOW_NORMAL)
    
    while True:
        display_frame = arrow_frames[curr_idx].copy()
                
        cv2.imshow('Optical Flow Visualization', display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('a') or key == ord('A'):
            curr_idx = max(curr_idx - 1, 0)
            log_idx()
        elif key == ord('d') or key == ord('D'):
            curr_idx = min(curr_idx + 1, len(arrow_frames) - 1)
            log_idx()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualise_flow_test(num_frames=150, frame_interval=30)