import cv2

from src.load_handler import load_vid1
from src.flow_handler import *
from src.logger import log

def calc_tiles(old_points, new_points):
    ROWS = 6
    COLS = 8
    tile_size = FlowConstants.TILE_SIZE
    
    tiles = np.zeros((ROWS, COLS))
    sums = np.zeros((ROWS, COLS))
    counts = np.zeros((ROWS, COLS))
    
    flow_vectors = new_points - old_points
    magnitudes = flow_magnitudes(flow_vectors)
    
    for i, (x, y) in enumerate(old_points):
        col = int(x) // tile_size
        row = int(y) // tile_size
        if row < ROWS and col < COLS:
            sums[row, col] += magnitudes[i]
            counts[row, col] += 1
    
    for row in range(ROWS):
        for col in range(COLS):
            if counts[row, col] > 0:
                tiles[row, col] = sums[row, col] / counts[row, col]
    
    return tiles

def normalize_to_depth(tiles):
    max_magnitude = np.max(tiles)
    if max_magnitude > 0:
        normalized = (tiles / max_magnitude) * 9
    else:
        normalized = tiles
    
    return normalized


motion_est = np.array([
    [9.0, 5.0, 5.8, 1.2, 1.1, 1.0, 2.1, 0.7],
    [3.3, 1.5, 1.7, 0.6, 0.5, 0.9, 1.1, 2.8],
    [3.6, 2.2, 1.7, 1.9, 1.8, 2.3, 3.1, 3.3],
    [4.0, 4.5, 2.8, 3.9, 3.5, 2.7, 4.5, 3.6],
    [0.0, 3.3, 0.0, 3.1, 2.7, 2.6, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.7, 2.3, 0.0, 0.0]
]).flatten()


def compare_tiles(tiles):
    tiles_normalized = normalize_to_depth(tiles)
    tiles = tiles_normalized.flatten()

    correlation = np.corrcoef(tiles, motion_est)[0, 1]
    covariance = np.cov(tiles, motion_est)[0, 1]

    return correlation, covariance


if __name__ == "__main__":
    frames = load_vid1(2)
    frame0 = frames[0]
    frame1 = frames[1]
    
    # Convert to grayscale
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)


    
    
    old, new = calc_points(gray0, gray1)
    tiles = calc_tiles(old, new)

    print("Actual tiles (raw):")
    print(tiles)
    print("\nNormalized tiles (0-9 scale):")
    print(normalize_to_depth(tiles))

    corr, cov = compare_tiles(tiles)
    
    log.info(corr)
    log.info(cov)
