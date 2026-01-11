import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ------------------ Page Config ------------------
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Shape & Contour Analyzer")

st.write("""
This application detects **geometric shapes**, counts objects,
and displays **area & perimeter** using **contour-based feature extraction**.
""")

# ------------------ Upload Image ------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert to OpenCV BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ------------------ Preprocessing ------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ------------------ Find Contours ------------------
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_count = 0
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignore noise
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        shape_name = "Unknown"

        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif len(approx) > 4:
            shape_name = "Circle"

        shape_count += 1

        # Draw contour and label
        cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 2)
        x, y = approx[0][0]
        cv2.putText(
            img_bgr,
            shape_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        results.append({
            "Shape": shape_name,
            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2)
        })

    # Convert back to RGB
    final_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ------------------ Display ------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🧠 Detected Shapes")
        st.image(final_img, use_container_width=True)

    st.success(f"🔢 Total Objects Detected: {shape_count}")

    st.subheader("📊 Shape Details")
    st.table(results)

else:
    st.info("Please upload an image to start analysis.")
