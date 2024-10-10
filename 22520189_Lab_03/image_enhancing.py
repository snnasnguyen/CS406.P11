import cv2
import numpy as np
import streamlit as st
from PIL import Image

def add_gaussian_noise(image, mean=0, std=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def load_image(image_file):
    img = Image.open(image_file)
    img = np.array(img.convert('RGB'))
    return img

def sharpen_image(image):
    # Sharpening kernel
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])

    kernel1 = np.array([[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]])
    # Apply the kernel to the image using cv2.filter2D
    sharpened = cv2.filter2D(image, -1, kernel)
    
    sharpened_1 = cv2.filter2D(image, -1, kernel1)
    
    #Apply 
    gaussian_blur = cv2.GaussianBlur(image,(5,5), 1)

    sharpening_2 = cv2.addWeighted(image, 3.5, gaussian_blur, -2.5, 0)
    sharpening_3 = cv2.addWeighted(image, 2.5, gaussian_blur, -1.5, 0)
    
    gaussian_blur_1 = cv2.GaussianBlur(image,(7,7), 3)
    
    sharpened_4 = cv2.addWeighted(image, 6.5, gaussian_blur_1, -5.5, 0)

    return sharpened, sharpened_1, sharpening_2, sharpening_3, sharpened_4

def edge_detection(image):

    # Sobel Edge Detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # Prewitt Edge Detection 
    kernel_prewitt_x = np.array([[ -1, 0, 1], 
                                [ -1, 0, 1], 
                                [ -1, 0, 1]])

    kernel_prewitt_y = np.array([[ -1, -1, -1], 
                                [  0,  0,  0], 
                                [  1,  1,  1]])

    prewitt_x = cv2.filter2D(image, -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_prewitt_y)
    prewitt_combined = cv2.add(prewitt_x, prewitt_y)

    # Canny Edge Detection
    canny_edges = cv2.Canny(image, 100, 200)

    return sobel_combined, prewitt_combined, canny_edges

st.set_page_config(layout="wide")

st.write("<div style='text-align: center; font-size:48px; font-weight: bold; padding-bottom: 16px'>Image Enhancing</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])


#Upload ảnh
if uploaded_file is not None:
    st.write("<div style='text-align: center; font-size:40px; font-weight: bold; padding-bottom: 48px'>Denoising:</div>", unsafe_allow_html=True)    
    img = load_image(uploaded_file)

    blurred_image = cv2.GaussianBlur(img, (15, 15), 0)

    #Noisy with Gaussian
    noisy_img = add_gaussian_noise(img)
    
    #Denoise with Mean
    mean_denoised = cv2.blur(noisy_img, ksize=(5, 5))
    
    #Denoise with Median
    median_denoised = cv2.medianBlur(noisy_img, ksize=5)

    bilateral_denoised = cv2.bilateralFilter(noisy_img, 9, 75, 75) 
        
    cols_denoised = st.columns(5)

    cols_denoised[0].image(img, caption="Original")
        
    cols_denoised[1].image(noisy_img, caption="Noisy(Gaussian)")
        
    cols_denoised[2].image(mean_denoised, caption="Denoised(Mean)")
        
    cols_denoised[3].image(median_denoised, caption="Denoised(Median)")
    
    cols_denoised[4].image(bilateral_denoised, caption='Denoised(Bilateral)')
        
    st.write("<div style='text-align: center; font-size:40px; font-weight: bold; padding-bottom: 48px'>Sharpening:</div>", unsafe_allow_html=True)    

    cols_sharpened = st.columns(5)

    sharpened, sharpened_1, sharpened_2, sharpened_3, sharpened_4 = sharpen_image(img)

    cols_sharpened[0].image(sharpened, caption='Sharpened(Kernel)')
    
    cols_sharpened[1].image(sharpened_1, caption='Sharpened(Kernel)')
    
    cols_sharpened[2].image(sharpened_2, caption='Sharpened(AddWeighted)')

    cols_sharpened[3].image(sharpened_3, caption='Sharpened(AddWeighted)')

    cols_sharpened[4].image(sharpened_4, caption='Sharpened(AddWeighted)')

    st.write("<div style='text-align: center; font-size:40px; font-weight: bold; padding-bottom: 48px'>Edge Detection Filter:</div>", unsafe_allow_html=True)    

    cols_edge = st.columns(4)
    
    sobel, prewitt, canny_edges = edge_detection(img)
    
    cols_edge[0].image(sharpened, caption='Original')
    
    cols_edge[1].image(sobel, caption='Sobel')
    
    cols_edge[2].image(prewitt, caption='Prewitt')
    
    cols_edge[3].image(canny_edges, caption='Canny')