import cv2
import numpy as np
import streamlit as st
import pickle
from PIL import Image

# Hàm xử lý ảnh
def process_image(image):
    # HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist0_hsv = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
    hist1_hsv = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])
    hist2_hsv = cv2.calcHist([image_hsv], [2], None, [256], [0, 256])
    input_hist_hsv = np.concatenate((hist0_hsv, hist1_hsv, hist2_hsv))
    cv2.normalize(input_hist_hsv, input_hist_hsv, 0, 1, cv2.NORM_MINMAX)
    
    # RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist0_rgb = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist1_rgb = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist2_rgb = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    input_hist_rgb = np.concatenate((hist0_rgb, hist1_rgb, hist2_rgb))
    cv2.normalize(input_hist_rgb, input_hist_rgb, 0, 1, cv2.NORM_MINMAX)
    
    return input_hist_hsv, input_hist_rgb

# Hàm tìm ảnh tương đồng (HSV)
def top_10_similar_hsv(img_hist, loaded_data, method=cv2.HISTCMP_CORREL):
    similar_images = []
    
    for path, hist in loaded_data:
        similarity = cv2.compareHist(img_hist, hist, method)
        similar_images.append((path, similarity))
    
    if (method == cv2.HISTCMP_BHATTACHARYYA or method == cv2.HISTCMP_CHISQR):
        similar_images.sort(key=lambda x: x[1], reverse=False)
    else: 
        similar_images.sort(key=lambda x: x[1], reverse=True)
    
    return similar_images[:10]

# Hàm tìm ảnh tương đồng (RGB)
def top_10_similar_rgb(img_hist, loaded_data, method=cv2.HISTCMP_CORREL):
    similar_images = []
    
    for path, hist in loaded_data:
        similarity = cv2.compareHist(img_hist, hist, method)
        similar_images.append((path, similarity))
    
    if (method == cv2.HISTCMP_BHATTACHARYYA or method == cv2.HISTCMP_CHISQR):
        similar_images.sort(key=lambda x: x[1], reverse=False)
    else: 
        similar_images.sort(key=lambda x: x[1], reverse=True)

    return similar_images[:10]

# Đọc hist đã lưu 
with open('data_hsv.pkl', 'rb') as file:
    loaded_data_hsv = pickle.load(file)
    
with open('data_rgb.pkl', 'rb') as file:
    loaded_data_rgb = pickle.load(file)

st.write("<div style='text-align: center; font-size:48px; font-weight: bold; padding-bottom: 16px'>Tìm ảnh tương đồng</div>", unsafe_allow_html=True)
st.write("Tải lên một ảnh để tìm 10 ảnh tương đồng nhất")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

# Danh sách lựa chọn phương pháp so sánh
method_list = {
    "Correlation": cv2.HISTCMP_CORREL,
    "Chi-Square": cv2.HISTCMP_CHISQR,
    "Intersection": cv2.HISTCMP_INTERSECT,
    "Bhattacharyya Distance": cv2.HISTCMP_BHATTACHARYYA
}

#Upload ảnh
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    st.image(uploaded_file, caption="Ảnh gốc", use_column_width="auto")
    
    input_hist_hsv, input_hist_rgb = process_image(image_cv)   
    
    # Hiển thị kết quả cho HSV
    st.write("<div style='text-align: center; font-size:24px; font-weight: bold; padding-bottom: 16px'>Top 10 ảnh tương đồng nhất (HSV):</div>", unsafe_allow_html=True)    
    
    method_hsv = st.selectbox("Chọn phương pháp so sánh cho HSV", list(method_list.keys()), index=0)
    top_10_similar_hsv_images = top_10_similar_hsv(input_hist_hsv, loaded_data_hsv, method=method_list[method_hsv])
    
    cols = st.columns(5)
    
    for i, (image_path, similarity) in enumerate(top_10_similar_hsv_images):
        image_result = Image.open(image_path)
        with cols[i % 5]:
            st.image(image_result)
            label = image_path.split('\\')[1]
            st.markdown(f"**Similarity:** {similarity:.2f}  \n **Label:** {label}")

    # Hiển thị kết quả cho RGB
    st.write("<div style='text-align: center; font-size:24px; font-weight: bold; padding-bottom: 16px'>Top 10 ảnh tương đồng nhất (RGB):</div>", unsafe_allow_html=True)
    
    method_rgb = st.selectbox("Chọn phương pháp so sánh cho RGB", list(method_list.keys()), index=0)
    top_10_similar_rgb_images = top_10_similar_rgb(input_hist_rgb, loaded_data_rgb, method=method_list[method_rgb])
    
    cols = st.columns(5)
    
    for i, (image_path, similarity) in enumerate(top_10_similar_rgb_images):
        image_result = Image.open(image_path)
        with cols[i % 5]:
            st.image(image_result)
            label = image_path.split('\\')[1]
            st.markdown(f"**Similarity:** {similarity:.2f}  \n **Label:** {label}")
