import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage import measure

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Shape & Contour Analyzer")

# ---------- Upload image ----------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ---------- Preprocessing ----------
    gray = rgb2gray(image_np)
    binary = gray < 0.8   # threshold

    # ---------- Find contours ----------
    contours = measure.find_contours(binary, 0.8)

    output = image.copy()
    draw = ImageDraw.Draw(output)

    shape_count = 0
    st.subheader("📊 Detected Shapes Details")

    for contour in contours:
        if len(contour) < 60:   # remove noise
            continue

        shape_count += 1

        # ---------- Feature Extraction ----------
        area = len(contour)

        perimeter = np.sum(
            np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        )

        # ---------- Shape Detection (Approximate) ----------
        if area < 400:
            shape = "Triangle / Small Shape"
        elif area < 900:
            shape = "Quadrilateral"
        else:
            shape = "Circle"

        # ---------- Draw Bounding Box ----------
        y_coords = contour[:, 0]
        x_coords = contour[:, 1]
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))

        draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=2)
        draw.text((min_x, min_y - 10), shape, fill="red")

        # ---------- Display info ----------
        st.write(f"**Shape {shape_count}: {shape}**")
        st.write(f"Area: {area} pixels")
        st.write(f"Perimeter: {int(perimeter)} pixels")
        st.write("---")

    # ---------- Display Images ----------
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(output, caption="Processed Image", use_column_width=True)

    st.success(f"✅ Total Shapes Detected: {shape_count}")
