import cv2
import os

from poses.get_skeleton import get_skeleton
import cv2
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt
import numpy as np
from playsound import playsound


folder = "pregnantyoga\\ExercisePoses"


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        print(filename)
        if img is not None:
            images.append(img)
    return images


def landmark_list(res):
    if res.pose_landmarks is None:
        return []
    return list(res.pose_landmarks.landmark)


def get_coord(l):
    coord = []
    for i in l:
        x = i.x
        y = i.y
        coord.append([x, y])
    return coord


def check_matching(im1, im2, threshold):
    res1 = get_skeleton(im1, True)
    l1 = landmark_list(res1)
    if not l1:
        return False
    coord1 = get_coord(l1)
    primary = np.array(coord1)

    res2 = get_skeleton(im2)
    l2 = landmark_list(res2)
    coord2 = get_coord(l2)
    secondary = np.array(coord2)

    n = primary.shape[0]
    def pad(x): return np.hstack([x, np.ones((x.shape[0], 1))])
    def unpad(x): return x[:, :-1]
    X = pad(primary)
    Y = pad(secondary)

    A, res, rank, s = np.linalg.lstsq(X, Y)

    def transform(x): return unpad(np.dot(pad(x), A))
    m = np.abs(secondary - transform(primary)).max()
    if m <= threshold:
        return True
    return False
