import numpy as np
import cv2

from src.load_handler import read_gt, read_imgs
from src.flow_handler import FlowConstants
from src.logger import log


def get_tracked_points(frame1, frame2):
    # Convert to uint8
    if frame1.dtype != np.uint8:
        gray1 = (frame1 * 255).astype(np.uint8)
        gray2 = (frame2 * 255).astype(np.uint8)
    else:
        gray1 = frame1
        gray2 = frame2
    
    points1 = cv2.goodFeaturesToTrack(gray1, FlowConstants.MAX_CORNERS, 
                                      FlowConstants.QUALITY_LEVEL, 
                                      FlowConstants.MIN_DISTANCE)
    
    if points1 is None:
        return [], [], []
    
    points2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, 
                                                      points1,
                                                      None,
                                                      winSize=FlowConstants.LK_WINSIZE,
                                                      maxLevel=FlowConstants.LK_MAX_LEVEL)    
    old_points = []
    new_points = []
    flow_vectors = []
    
    h, w = gray1.shape
    
    for i, (new, old, stat) in enumerate(zip(points2, points1, status)):
        if stat[0] == 1:
            x1, y1 = int(old[0][0]), int(old[0][1])
            x2, y2 = int(new[0][0]), int(new[0][1])
            
            # Skip if points are outside image bounds
            if x1 < 0 or x1 >= w or y1 < 0 or y1 >= h:
                continue
            if x2 < 0 or x2 >= w or y2 < 0 or y2 >= h:
                continue
            
            # Skip if flow vector is unreasonably large (> 50 pixels)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) > 50 or abs(dy) > 50:
                continue
            
            old_points.append((x1, y1))
            new_points.append((x2, y2))
            flow_vectors.append([dx, dy])
            
    return old_points, flow_vectors, len(old_points)
    
    
def calc_AE_at_points(flow_vectors, gt_flow, points):
    ae_values = []
    for i, (u, v) in enumerate(flow_vectors):
        x, y = points[i]
        # Get ground truth flow at this point
        gt_u = gt_flow[y, x, 0]
        gt_v = gt_flow[y, x, 1]
        
        num = u * gt_u + v * gt_v + 1.0
        den1 = np.sqrt(u**2 + v**2 + 1.0)
        den2 = np.sqrt(gt_u**2 + gt_v**2 + 1.0)
        cos_a = num / (den1 * den2 + 0.00001)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        
        ae = np.arccos(cos_a) * 180.0 / np.pi
        ae_values.append(ae)
    
    return np.array(ae_values)


def calc_EPE_at_points(flow_vectors, gt_flow, points):
    epe_values = []
    h, w = gt_flow.shape[:2]
    
    for i, (u, v) in enumerate(flow_vectors):
        x, y = points[i]
        
        # Skip if point is outside image bounds
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        
        # Skip if flow vector is unreasonably large (> 100 pixels)
        if abs(u) > 100 or abs(v) > 100:
            continue
        
        gt_u = gt_flow[y][x][0]
        gt_v = gt_flow[y][x][1]
        
        du = u - gt_u
        dv = v - gt_v
        epe = np.sqrt(du**2 + dv**2)
        
        # Skip if EPE is unreasonably large (> 50 pixels)
        if epe > 50:
            continue
            
        epe_values.append(epe)
    
    if len(epe_values) == 0:
        return np.array([0])
    
    return np.array(epe_values)


def test_AE(frames_path, flow_path):
    log.mark(f"Testing Angular Error on: {frames_path}")
    imgs = read_imgs(frames_path)
    gt = read_gt(flow_path)
    
    all_ae = []
    total_points = 0
    
    for i in range(len(imgs)-1):
        points, flow_vectors, num_points = get_tracked_points(imgs[i], imgs[i+1])
        
        if num_points == 0:
            log.warning(f"  Frame {i}->{i+1}: No points tracked")
            continue
        
        ae_map = calc_AE_at_points(flow_vectors, gt, points)
        avg_ae = np.mean(ae_map)
        all_ae.append(avg_ae)
        total_points += num_points
        log.info(f"  Frame {i}->{i+1}: {avg_ae:.4f} degrees ({num_points} points)")
    
    if all_ae:
        avg_total = np.mean(all_ae)
        log.success(f"  Average AE: {avg_total:.4f} degrees over {total_points} total points")
        return avg_total
    else:
        log.warning("No valid AE results")
        return -1


def test_EPE(frames_path, flow_path):
    log.mark(f"Testing End-Point Error on: {frames_path}")
    imgs = read_imgs(frames_path)
    gt = read_gt(flow_path)
    
    all_epe = []
    total_points = 0
    
    for i in range(len(imgs)-1):
        points, flow_vectors, num_points = get_tracked_points(imgs[i], imgs[i+1])
        
        if num_points == 0:
            log.warning(f"  Frame {i}->{i+1}: No points tracked")
            continue
        
        epe_map = calc_EPE_at_points(flow_vectors, gt, points)
        avg_epe = np.mean(epe_map)
        all_epe.append(avg_epe)
        total_points += num_points
        log.info(f"  Frame {i}->{i+1}: {avg_epe:.4f} pixels ({num_points} points)")
    
    if all_epe:
        avg_total = np.mean(all_epe)
        log.success(f"  Average EPE: {avg_total:.4f} pixels over {total_points} total points")
        return avg_total
    else:
        log.warning("No valid EPE results")
        return -1


if __name__ == "__main__":
    # Use the correct paths relative to the data folder
    set_name = "RubberWhale"
    
    # The read_imgs and read_gt functions already look in DATA_PATH
    # Just pass the subfolder names
    frames_path = f"MB-frames/{set_name}"
    flow_path = f"MB-flow/{set_name}"
    
    log.mark(f"Testing {set_name}")
    
    ae = test_AE(frames_path, flow_path)
    epe = test_EPE(frames_path, flow_path)
    
    if ae != -1 and epe != -1:
        log.success(f"Final Results for {set_name}:")
        log.success(f"Angular Error (AE): {ae:.4f} degrees")
        log.success(f"End-Point Error (EPE): {epe:.4f} pixels")