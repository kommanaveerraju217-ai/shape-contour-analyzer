import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Shape & Contour Analyzer")

# ---------- Helper function to calculate side lengths ----------
def side_lengths(pts):
    lengths = []
    for i in range(4):
        p1 = pts[i][0]
        p2 = pts[(i + 1) % 4][0]
        length = np.linalg.norm(p1 - p2)
        lengths.append(length)
    return lengths

# ---------- Upload image ----------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # ---------- Preprocessing ----------
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ---------- Find contours ----------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image_np.copy()
    shape_count = 0

    st.subheader("📊 Detected Shapes Details")

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore small noise
        if area < 500:
            continue

        shape_count += 1
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        vertices = len(approx)

        # ---------- Shape Detection ----------
        if vertices == 3:
            shape = "Triangle"

        elif vertices == 4:
            sides = side_lengths(approx)
            tolerance = 10

            if (abs(sides[0] - sides[1]) < tolerance and
                abs(sides[1] - sides[2]) < tolerance and
                abs(sides[2] - sides[3]) < tolerance):
                shape = "Square"

            elif (abs(sides[0] - sides[2]) < tolerance and
                  abs(sides[1] - sides[3]) < tolerance):
                shape = "Rectangle"

            else:
                shape = "Quadrilateral"

        elif vertices > 4:
            shape = "Circle"

        else:
            shape = "Unknown"

        # ---------- Draw contour & label ----------
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(output, shape, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ---------- Display info ----------
        st.write(f"**Shape {shape_count}: {shape}**")
        st.write(f"Area: {int(area)} pixels")
        st.write(f"Perimeter: {int(perimeter)} pixels")
        st.write("---")

    # ---------- Display Images ----------
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(output, caption="Processed Image", use_column_width=True)

    st.success(f"✅ Total Shapes Detected: {shape_count}")
