import cv2
from src.load_handler import load_vid1
from src.logger import log
from src.save_handler import save_depth_frame
from src.depth_analyzer import get_depth_map, draw_depth, init_max_magn

def main():
    frames = load_vid1()
    global_max = init_max_magn(frames[0], frames[1])

    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    
    curr_idx = 0 
    while True:
        curr = frames[curr_idx]
        nxt = frames[curr_idx + 1]
        
        points, magnitudes = get_depth_map(curr, nxt)
        
        if points is not None and len(points) > 0:
            display_frame = draw_depth(curr, points, magnitudes)
        else:
            display_frame = curr
        
        save_depth_frame(f"frame_{curr_idx:04d}.png", display_frame)
        cv2.imshow('Depth Map', display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('a') or key == ord('A'):
            curr_idx = max(curr_idx - 1, 0)
            log.info(f"Frame {curr_idx + 1}")
        elif key == ord('d') or key == ord('D'):
            curr_idx = min(curr_idx + 1, len(frames) - 2)
            log.info(f"Frame {curr_idx + 1}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
