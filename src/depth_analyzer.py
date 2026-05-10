import numpy as np
import cv2
from src.flow_handler import calc_points

COLORS = [
    [255, 0, 255],
    [255, 0, 128],
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85]
]

MIN_RADIUS = 2
MAX_RADIUS = 8

GLOBAL_MAX_MAG = None

def init_max_magn(frame1, frame2):
    global GLOBAL_MAX_MAG
    p1, p2 = calc_points(frame1, frame2)
    if p1 is not None and len(p1) > 0:
        magnitudes = np.sqrt(np.sum((p2 - p1)**2, axis=1))
        GLOBAL_MAX_MAG = np.percentile(magnitudes, 95)
        if GLOBAL_MAX_MAG == 0:
            GLOBAL_MAX_MAG = 1.0
    return GLOBAL_MAX_MAG

def get_color(magnitude):
    global GLOBAL_MAX_MAG
    if GLOBAL_MAX_MAG is None:
        return COLORS[0]
    ratio = magnitude / GLOBAL_MAX_MAG
    if ratio > 1.0:
        ratio = 1.0
    index = int(ratio * (len(COLORS) - 1))
    return COLORS[index]

def get_depth_map(frame1, frame2):
    p1, p2 = calc_points(frame1, frame2)
       
    magnitudes = np.sqrt(np.sum((p2 - p1)**2, axis=1))
    
    return p2, magnitudes

def draw_depth(frame, points, magnitudes):
    if points is None or len(points) == 0:
        return frame.copy()
    
    copy_frame = frame.copy()
    
    for point, mag in zip(points, magnitudes):
        x, y = int(point[0]), int(point[1])
        color = get_color(mag)
        
        radius = int(mag * 10) + 2
        if radius < MIN_RADIUS:
            radius = MIN_RADIUS
        if radius > MAX_RADIUS:
            radius = MAX_RADIUS
        
        cv2.circle(copy_frame, (x, y), radius, color, -1)
        cv2.circle(copy_frame, (x, y), radius, (0, 0, 0), 1)
    
    return copy_frame
