#!/usr/bin/env python3
import time
import os
import matplotlib.pyplot as plt
from src.flow_handler import *
from src.logger import log
from src.load_handler import load_test_frames

SAVE_DIR = './data/fps/'

def main():
    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    NR_TEST = 180
    frames = load_test_frames(NR_TEST)
    fps_values = []

    for i in range(len(frames) - 1):
        start = time.perf_counter()
        calc_flow(frames[i], frames[i + 1])
        elapsed = time.perf_counter() - start
        
        fps = 1.0 / elapsed
        fps_values.append(fps)
        
        if (i + 1) % 50 == 0:
            log.info(f"Processed {i + 1}/{len(frames) - 1} frames")
    
    # Calculate statistics
    avg_fps = sum(fps_values) / len(fps_values)
    
    log.info(f"\nResults:")
    log.info(f"  Total frame pairs: {len(fps_values)}")
    log.info(f"  Average FPS: {avg_fps:.2f}")
    log.info(f"  Min FPS: {min(fps_values):.2f}")
    log.info(f"  Max FPS: {max(fps_values):.2f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(fps_values)), fps_values, 'b-', linewidth=1)
    plt.axhline(y=avg_fps, color='r', linestyle='--', label=f'Average: {avg_fps:.2f} FPS')
    plt.xlabel('Frame Pair Number')
    plt.ylabel('FPS')
    plt.title('FPS Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = SAVE_DIR + f"fps_over_time_{int(time.time())}.png"
    plt.savefig(path, dpi=150)
    log.info(f"Plot saved to: {path}")
    plt.show()


if __name__ == "__main__":
    main()