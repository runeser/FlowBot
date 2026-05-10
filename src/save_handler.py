import cv2
import os

# BASE_PATH = os.path.abspath("./../data/")

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(FILE_DIR)
BASE_PATH = os.path.join(FILE_DIR, "../data/")

def save_plot(name, plot):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)


def save_img_data(subpath, img):
    path = BASE_PATH + subpath
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(path, img)
    return path

ARROW_PATH = BASE_PATH + "arrows/"
def save_arrow_frame(img_name, arrow_frame):
    os.makedirs(ARROW_PATH, exist_ok=True)
    PATH = ARROW_PATH + img_name
    cv2.imwrite(PATH, arrow_frame)
    return PATH

PATH_3D = BASE_PATH + "depth/"
def save_depth_frame(subpath, img):
    path = PATH_3D + subpath
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    return path

METRICS_PATH = BASE_PATH + "metrics/"
def save_metrics(metrics, filename="metrics.txt"):
    os.makedirs(METRICS_PATH, exist_ok=True)
    PATH = METRICS_PATH + filename
    with open(PATH, 'w') as f:
        f.write(metrics)
    return PATH




# def save_3D_points(subpath, points3d):
#     path = PATH_3D + subpath
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     np.save(path, points3d)
#     return path