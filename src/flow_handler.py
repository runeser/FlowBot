import cv2
import numpy as np

class FlowConstants:
    MAX_CORNERS = 100000
    BLOCK_SIZE = 11
    MIN_DISTANCE = 6
    QUALITY_LEVEL = 0.01
      
    LK_WINSIZE = (15, 15)
    LK_MAX_LEVEL = 2
    LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    
    TILE_SIZE = 80


"""
https://docs.opencv.org/4.x/db/d7f/tutorial_js_lucas_kanade.html
"""

def clean_frame(frame):
    if frame is None or frame.size == 0:
        return None
    
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame

def get_gray_frames( frame1, frame2):
    gray1 = clean_frame(frame1)
    gray2 = clean_frame(frame2)

    return gray1, gray2

def detect_features( gray_frame):
    points = cv2.goodFeaturesToTrack(
        gray_frame, 
        FlowConstants.MAX_CORNERS, 
        FlowConstants.QUALITY_LEVEL, 
        FlowConstants.MIN_DISTANCE
    )
    return points

def track_features( gray1, gray2, points1):
    points2, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, 
        points1, None,
        winSize=FlowConstants.LK_WINSIZE,
        maxLevel=FlowConstants.LK_MAX_LEVEL
    )
    return points2, status

def filter_successful( points1, points2, status):
    success_mask = (status == 1).flatten()
    if not success_mask.any():
        return None, None
    
    filtered_p1 = points1[success_mask].reshape(-1, 2)
    filtered_p2 = points2[success_mask].reshape(-1, 2)
    
    return filtered_p1, filtered_p2

def calc_points( frame1, frame2):
    if frame1 is None or frame2 is None:
        return None, None
    
    gray1, gray2 = get_gray_frames(frame1, frame2)
    points1 = detect_features(gray1)
    
    if points1 is None or len(points1) == 0:
        return None, None
    
    points2, status = track_features(gray1, gray2, points1)
    good_p1, good_p2 = filter_successful(points1, points2, status)
    return good_p1, good_p2


def calc_flow( frame1, frame2):
    p1, p2 = calc_points(frame1, frame2)
    
    if p1 is None: return np.array([]) 
    if p2 is None: return np.array([])


    return (p2 - p1).astype(np.float32)


def get_magnitudes(points):
    dxs = points[:, 0]
    dys = points[:, 1]
    return np.sqrt(dxs**2 + dys**2)

def flow_magnitudes(flow):    
    magnitudes = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
    return magnitudes


def draw_arrows( frame, old_points, new_points):
    result = frame.copy()
    color = (0, 255, 0)  # Green 
    for (x1, y1), (x2, y2) in zip(old_points, new_points):
        cv2.arrowedLine(result, (int(x1), int(y1)), (int(x2), int(y2)), color, tipLength=0.3)
    
    return result

