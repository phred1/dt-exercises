#!/usr/bin/env python3

import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import matplotlib.pyplot as plt
import cv2
import logging
import random

DATASET_DIR="../../dataset/sim_data"

npz_index = 0

def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1



def remove_small_pixel_groups(input_im, h,w, gray_treshold, id_image):
    # cv2.imshow"input" + id_image, cv2.cvtColor(input_im, cv2.COLOR_RGB2BGR))
    img_bw = 255*(cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY) > gray_treshold).astype('uint8')
    # cv2.imshow"bw" + id_image, cv2.cvtColor(input_im, cv2.COLOR_RGB2GRAY))
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (h,w))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (h,w))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    mask = np.dstack([mask, mask, mask]) / 255
    # cv2.imshow"mask" + id_image, mask)
    out = input_im * mask
    out = out.astype(np.float32)
    out /= 255.

    return out

def remove_white_yellow_lines(input_img):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(255*gray, 170, 255, cv2.THRESH_BINARY)
    input_img *= 255
    input_img[thresh > 200] = 0
    input_img = input_img.astype(np.float32)
    input_img /= 255.
    return input_img


RGB_COLORS ={
    "1": [100,117, 226],
    "3": [116,114, 117],
    "2": [226, 111, 101],
    "0": [255,0,255],
    "4":[216,171,15]
}

def keep_only(im, object_type):
    copy = np.copy(im)
    # Make mask of all perfectly red pixels
    Rmask = np.all(copy == RGB_COLORS[object_type],  axis=-1)

    # Make all non-red pixels black
    copy[~Rmask] = [0,0,0]
    # cv2.imshowobject_type, cv2.cvtColor(copy, cv2.COLOR_RGB2BGR))
    return copy

def convert_to_binary(im, object_type):
    copy = np.copy(im)
    
    # Make mask of all perfectly red pixels
    Rmask = np.all(copy == RGB_COLORS[object_type],  axis=-1)

    # # Make all non-red pixels black
    copy[~Rmask] = [0,0,0]
    copy[Rmask] = [255,255,255]
    # cv2.imshow"binary " + object_type, cv2.cvtColor(copy, cv2.COLOR_RGB2BGR))
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return binary

def min_area(contour):
    return cv2.contourArea(contour) > 4

def contour(img, object_type):
    binary_img = convert_to_binary(img, object_type)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if min_area(c)] 
    cv2.drawContours(img, contours, -1, (255,0,0), 3)

    bound_rect = []
    labels = []
    for i, cnt in enumerate(contours):
        color = (0,255,0)
        bound_rect.append(bounding_boxes(cnt))
        labels.append(int(object_type))
        cv2.rectangle(img, (int(bound_rect[i][0]), int(bound_rect[i][1])), \
            (int(bound_rect[i][2]), int(bound_rect[i][3])), color, 2)
        
    
    return bound_rect, labels

def bounding_boxes(countour):
    xmin, ymin, w, h = cv2.boundingRect(countour)
    xmax = xmin + w
    ymax = ymin + h
    return xmin,ymin,xmax,ymax 

def clean_segmented_image(seg_img):
    # cv2.imshow("input", seg_img)
    # cv2.imshow("segmented", cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    out = remove_small_pixel_groups(seg_img, 7, 7, 20,"1")
    # cv2.imshow"out1", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    out = remove_white_yellow_lines(out)
    # cv2.imshow"out2", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    out = 255 * out
    out = out.astype("uint8")
    cv2.imwrite("test.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    all_bound_rects, all_labels = [], []
    duckies = keep_only(out, "1")
    rects, labels = contour(duckies, "1")
    all_bound_rects = all_bound_rects + rects
    all_labels = all_labels + labels

    buses = keep_only(out, "4")
    rects, labels = contour(buses, "4")
    all_bound_rects = all_bound_rects + rects
    all_labels = all_labels + labels

    trucks = keep_only(out, "3")
    rects, labels = contour(trucks, "3")
    all_bound_rects = all_bound_rects + rects
    all_labels = all_labels + labels

    cones = keep_only(out, "2")
    rects, labels = contour(cones, "2")
    all_bound_rects = all_bound_rects + rects
    all_labels = all_labels + labels

    sky = keep_only(out, "0")
    rects, labels = contour(sky, "0")
    all_bound_rects = all_bound_rects + rects
    all_labels = all_labels + labels

    cv2.waitKey(0)

    return all_bound_rects, all_labels

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500
while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)
        resized = cv2.resize(segmented_obs, (224,224), interpolation = cv2.INTER_AREA)
        bounding_rects, labels = clean_segmented_image(resized)
        chance = random.randint(0,100)
        resized_obs = cv2.resize(obs, (224,224), interpolation = cv2.INTER_AREA)
        save_npz(resized_obs, bounding_rects, labels)
        nb_of_steps += 1
        if done or nb_of_steps > MAX_STEPS:
            break