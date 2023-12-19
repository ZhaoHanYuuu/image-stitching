"""
main中用到的一些函数
"""
import cv2
import numpy as np


def compute_ratio_V(left, right):
    right_hsv = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
    right_v = computeV(right_hsv)
    left_hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
    left_v = computeV(left_hsv)
    bright_k = left_v / right_v
    return bright_k


def computeV(hsv_img):
    h, s, v = cv2.split(hsv_img)
    return np.sum(v)
