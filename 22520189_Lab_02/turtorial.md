<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS406.M11 - XỬ LÝ ẢNH VÀ ỨNG DỤNG</b></h1>
<h1 align="center"><b>IMAGE PROCESSING AND APPLICATIONS</b></h1>
<h2 align="center"><b>LAB 02</b></h2>

# Introduction
In this lab, I used histogram features in two color spaces: RGB and HSV. Then, I identified the top 10 most similar images using different comparison methods.
# Quickstart
## Download dataset
Link : <a href="https://drive.google.com/file/d/1F6sPtl0H-Sh7XPrAojDKcz_rBoUl_fgu/view?usp=sharing">Dataset Lab2</a>
## Install requirements
Open terminal and run
```bash
 pip install -r requirements.txt
```
## Calculate hist
Run `calHist.py` to create two pickle files: `data_hsv.pkl` and `data_rgb.pkl`
```bash
 python calHist.py
```
## Run streamlit
After creating the histogram data, use the terminal and run streamlit
```bash
 streamlit run find_top10_similarity.py
```
## Demo
<img src="https://raw.githubusercontent.com/snnasnguyen/CS406.P11/refs/heads/master/22520189_Lab_02/turtorial.gif" alt="Turtorial"></img>

