import os
import numpy as np
import cv2
from src.logger import log

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data") + "/"

def load_frames(path, nr_frames):
    # frames = np.array([])
    frames  = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.warning(f"Cannot open video: {path}")
        return frames
    
    for _ in range(nr_frames):
        ret, fr = cap.read()
        if not ret:  break
        frames.append(fr)
        
    cap.release()

    frames = np.array(frames)
    return frames


def rotate_180(frame):
    return cv2.rotate(frame, cv2.ROTATE_180)

# vid1_name = "prof_test1.h264"
vid1_name = "vid1.mp4"
VID1_PATH = DATA_PATH + vid1_name

def load_vid1(nr_frames=480):
    log.info(f"Loading from: {VID1_PATH}")
    frames = load_frames(VID1_PATH, nr_frames)

    if vid1_name == "prof_test1.h264":
        log.info("Rotating frames 180 degrees")
        rotated_frames = []
        for frame in frames:
            rotated_frames.append(rotate_180(frame))
        frames = np.array(rotated_frames)

    log.info(f'VID1: loaded {len(frames)} frames')
    return frames


vid2_name = "prof_test2.h264"
# vid2_name = "vid2.mp4"
VID2_PATH = DATA_PATH + vid2_name
def load_vid2(nr_frames=480):
    frames = load_frames(VID2_PATH, nr_frames)

    if (vid2_name == "prof_test2.h264"):
        log.info("Rotating frames 180 degrees")
        rotated_frames = []
        for frame in frames:
            rotated_frames.append(rotate_180(frame))
        frames = np.array(rotated_frames)

    log.info(f'VID2: loaded {len(frames)} frames')
    return frames

def load_test_frames(num_frames=100):
    frames = load_vid1(num_frames)
    return frames



IMGS_PATH = DATA_PATH
def read_imgs(subpath):
    path = IMGS_PATH + subpath
    imgs = []
    png_files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    for file in png_files:
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imgs.append(img.astype(np.float32) / 255.0)
    
    return imgs

GT_PATH = DATA_PATH 
def read_gt(subpath):
    path = GT_PATH +subpath
    flow_file = os.path.join(path, 'flow10.flo')
    with open(flow_file, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        gt = np.resize(data, (h, w, 2))
    return gt



if __name__ == "__main__":
    load_vid1()
    load_vid2()
    load_test_frames()
    #read_gt()
    #read_imgs()