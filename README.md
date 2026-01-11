****Shape & Contour Analyzer (Streamlit App)****
An interactive computer vision dashboard built using Streamlit and OpenCV to detect geometric shapes, count objects, and compute their area and perimeter using contour-based feature extraction.

 ****Problem Description****
Manual identification and measurement of geometric objects in images is time-consuming and error-prone.
This project automates the process of detecting shapes, counting objects, and extracting geometric features using image processing techniques.

****Objectives****
1. Detect common geometric shapes (Triangle, Square, Rectangle, Circle)

2. Count the number of objects in an image.

3. Calculate area and perimeter of each detected object

4. Build an interactive dashboard using Streamlit

****How to Run****

Install Dependencies (if not already installed):
```
pip install streamlit opencv-python-headless numpy pandas pillow

```
Run the App:
```
streamlit run app.py

```
Open in Browser: The app will usually open automatically at http://localhost:8501.

****Features****

Dashboard 

1.Upload Image: Supports JPG, PNG.

2.Original vs Processed: Compare the raw input with the detected contours.

3.Metrics: View total object count and a data table with:

Shape Type (Triangle, Square, Rectangle, Circle, Polygon)

Area (px)

Perimeter (px)

****Technologies Used****

Python

Streamlit – Web application framework

OpenCV – Image processing & contour detection

NumPy – Numerical operations

Pillow (PIL) – Image handling
