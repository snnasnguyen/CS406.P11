import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit 
from tqdm import tqdm
import pickle

data_dir = './dataset/seg'
hists_hsv = []
hists_rgb = []
for subdir, dirs, files in os.walk(data_dir):
  for file in tqdm(files):
    image_path = os.path.join(subdir, file)
    image = cv2.imread(image_path)
    
    # HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist0_hsv = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
    hist1_hsv = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])
    hist2_hsv = cv2.calcHist([image_hsv], [2], None, [256], [0, 256])
    hist_hsv = np.concatenate((hist0_hsv, hist1_hsv, hist2_hsv))
    cv2.normalize(hist_hsv, hist_hsv, 0, 1, cv2.NORM_MINMAX)  
    hists_hsv.append((image_path, hist_hsv))
    
    # RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist0_rgb = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist1_rgb = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist2_rgb = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    hist_rgb = np.concatenate((hist0_rgb, hist1_rgb, hist2_rgb))
    cv2.normalize(hist_rgb, hist_rgb, 0, 1, cv2.NORM_MINMAX)  
    hists_rgb.append((image_path, hist_rgb))
    
with open('data_hsv.pkl', 'wb') as file:
    pickle.dump(hists_hsv, file)
    
with open('data_rgb.pkl', 'wb') as file:
    pickle.dump(hists_rgb, file)