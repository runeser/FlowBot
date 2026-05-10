import cv2
import numpy as np
import os
import time
from datetime import datetime
from src.flow_handler import FlowConstants, calc_flow
from src.logger import log
from src.load_handler import load_test_frames

VIDEO_PATH = 'prof_test1.h264'
SKIP_FRAMES = 150
TEST_FRAMES = 100
PERF_ITERATIONS = 25

SAVE_DIR = f"./metrics/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

TEST_VALUES = {
    'LK_MAX_LEVEL': [0, 1, 2, 3],
    'LK_WINSIZE': [(9,9), (11,11), (15,15), (21,21)],
    'MAX_CORNERS': [80, 100, 120, 150, 200],
    'QUALITY_LEVEL': [0.01, 0.02, 0.03, 0.05],
    'MIN_DISTANCE': [10, 15, 20, 25]
}

MIN_GOOD_MOTION = 0.5
MAX_GOOD_MOTION = 8

def calculate_metrics(flow_vectors, processing_time=None):
    if flow_vectors is None or len(flow_vectors) == 0:
        return {
            'num_points': 0,
            'tracking_quality': 0,
            'quality_score': 0,
            'fps': 0 if processing_time else None,
            'combined_score': 0
        }
    
    magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
    good_motion = (magnitudes > MIN_GOOD_MOTION) & (magnitudes < MAX_GOOD_MOTION)
    tracking_quality = np.mean(good_motion) if len(magnitudes) > 0 else 0
    
    metrics = {
        'num_points': len(flow_vectors),
        'tracking_quality': tracking_quality,
        'quality_score': tracking_quality * len(flow_vectors),
    }
    
    if processing_time and processing_time > 0:
        metrics['fps'] = 1.0 / processing_time
        metrics['combined_score'] = metrics['quality_score'] * metrics['fps']
    else:
        metrics['fps'] = 0
        metrics['combined_score'] = metrics['quality_score']
    
    return metrics

def measure_performance(frame_sequence, param_name, param_value):
    if len(frame_sequence) < 2:
        return 0, 0
    try:
        setattr(FlowConstants, param_name, param_value)
    except AttributeError:
        log.error(f"FlowConstants has no attribute {param_name}")
        return 0, 0
    
    total_time = 0
    valid_runs = 0
    quality_scores = []
    
    max_pairs = min(PERF_ITERATIONS, len(frame_sequence) - 1)
    
    for i in range(max_pairs):
        try:
            start = time.time()
            flow = calc_flow(frame_sequence[i], frame_sequence[i + 1])
            elapsed = time.time() - start
            
            if flow is not None and len(flow) > 0:
                total_time += elapsed
                valid_runs += 1
                metrics = calculate_metrics(flow, elapsed)
                quality_scores.append(metrics['quality_score'])
        except Exception as e:
            log.warning(f"Frame {i} failed: {e}")
    
    if valid_runs == 0:
        return 0, 0
    
    avg_time = total_time / valid_runs
    fps = 1.0 / avg_time
    avg_quality = np.mean(quality_scores) if quality_scores else 0
    
    return fps, avg_quality

def test_parameter(frame_sequence, param_name):
    log.mark(f"Testing {param_name}")
    original_value = getattr(FlowConstants, param_name)
    results = []
    values = TEST_VALUES[param_name]
    
    for value in values:
        fps, quality = measure_performance(frame_sequence, param_name, value)
        combined = fps * quality
        
        results.append({
            'value': value,
            'fps': fps,
            'quality_score': quality,
            'combined_score': combined
        })
        
        log.info(f"  {param_name}={value}: fps={fps}, quality={quality}, combined={combined}")
    
    setattr(FlowConstants, param_name, original_value)
    
    best = max(results, key=lambda x: x['combined_score'])
    log.success(f"Best {param_name}={best['value']} (fps={best['fps']}, quality={best['quality_score']})\n")
    
    txt_path = f"{SAVE_DIR}/{param_name}.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Parameter: {param_name}\n")
        f.write(f"Original value: {original_value}\n")        
        for r in results:
            f.write(f"Value: {r['value']}\n")
            f.write(f"  FPS: {r['fps']}\n")
            f.write(f"  Quality: {r['quality_score']}\n")
            f.write(f"  Combined: {r['combined_score']}\n\n")
        
        f.write(f"Best value: {best['value']}\n")
        f.write(f"Best FPS: {best['fps']}\n")
        f.write(f"Best quality: {best['quality_score']}\n")
    
    return best

def create_report(results_dict):
    report_path = f"{SAVE_DIR}/optimization_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for param_name, best in results_dict.items():
            f.write(f"{param_name}: {best['value']}\n")
            f.write(f"  FPS: {best['fps']}\n")
            f.write(f"  Quality: {best['quality_score']}\n\n")
        
        for param_name, best in results_dict.items():
            f.write(f"FlowConstants.{param_name} = {best['value']}\n")
    
    log.success(f"Report saved to {report_path}")

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    log.info(f"Saving metrics to {SAVE_DIR}")
    log.info("Loading test frames...")
    
    try:
        frame_sequence = load_test_frames()
        results_dict = {}
        
        for param_name in TEST_VALUES.keys():
            results_dict[param_name] = test_parameter(frame_sequence, param_name)
        
        create_report(results_dict)
        log.success(f"All tests completed! Results saved to {SAVE_DIR}")
        
    except Exception as e:
        log.error(f"Test failed: {e}")
