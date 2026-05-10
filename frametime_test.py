import cv2
import time
import statistics
import os
import matplotlib.pyplot as plt
from src.flow_handler import *
from src.load_handler import load_vid1
from src.logger import log


SAVE_DIR = './data/frametimes/'

def main():
    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    frames = load_vid1()
    
    if len(frames) < 2:
        log.error(f"Only got {len(frames)} frames")
        return
    
    frame_times_ms = []
    nr_frames = len(frames)
    for i in range(nr_frames-1):
        start = time.time()
        calc_flow(frames[i], frames[i + 1])
        elapsed_ms = (time.time() - start) * 1000
        
        frame_times_ms.append(elapsed_ms)
        
        if (i + 1) % 50 == 0:
            log.info(f"Processed {i + 1}/{len(frames) - 1} frames")
    
    # Calculate statistics
    avg_time = sum(frame_times_ms) / len(frame_times_ms)
    
    log.info(f"\nResults:")
    log.info(f"  Total frame pairs: {len(frame_times_ms)}")
    log.info(f"  Average frame time: {avg_time:.2f} ms")
    log.info(f"  Min frame time: {min(frame_times_ms):.2f} ms")
    log.info(f"  Max frame time: {max(frame_times_ms):.2f} ms")
    
    if len(frame_times_ms) > 1:
        log.info(f"  Std deviation: {statistics.stdev(frame_times_ms):.2f} ms")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(frame_times_ms)), frame_times_ms, 'b-', linewidth=1)
    plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Average: {avg_time:.2f} ms')
    plt.xlabel('Frame Pair Number')
    plt.ylabel('Processing Time (ms)')
    plt.title('Frame Time Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = SAVE_DIR + f"frametimes_{int(time.time())}.png"
    plt.savefig(path, dpi=150)
    log.info(f"Plot saved to: {path}")
    plt.show()


if __name__ == "__main__":
    main()