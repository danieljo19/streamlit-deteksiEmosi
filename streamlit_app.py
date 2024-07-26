import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
from datetime import datetime

st.set_page_config(
        page_title="DETEKSI EMOSI SISWA",
)

# Parameter HOG
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
orientations = 8

# Fungsi untuk mengekstraksi fitur HOG dari gambar berwarna
def extract_hog_features(image):
    img_resized = cv2.resize(image, (64, 128))
    img_gray = color.rgb2gray(img_resized)
    h_features = hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
    return h_features

# Load model PCA dan SVM
pca = joblib.load('pca_model_new17072024_pca_new.pkl')
svm = joblib.load('svm_model_new17072024new_c=10_pca_new.pkl')

# Inisialisasi detektor wajah Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk melakukan deteksi dan prediksi emosi pada wajah
def detect_and_predict_emotion(image):
    detected_faces = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        features = extract_hog_features(face)
        if features is not None:
            # Transformasi fitur menggunakan PCA
            pca_features = pca.transform([features])
            # Prediksi menggunakan SVM
            emotion_idx = svm.predict(pca_features)[0]
            emotion = label_encoder.inverse_transform([emotion_idx])[0]
            detected_faces.append((x, y, w, h, emotion))

            # Gambar bounding box dan label emosi pada wajah
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image, detected_faces

# Encode labels menjadi angka (konsisten dengan label yang digunakan saat pelatihan)
label_encoder = LabelEncoder()
emotions = ['anger', 'happy', 'sad', 'neutral']
label_encoder.fit(emotions)

# Streamlit UI
st.title("Sistem Deteksi Emosi Siswa")

# Tabs
tab1, tab2, tab3 = st.tabs(["Upload Video", "Preview Frame", "Export Data"])

# Tab 1: Upload Video
with tab1:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    # Pilih video default jika tidak ada video yang diupload
    if uploaded_video is None:
        default_video_path = 'VID_20240724_104550.mp4'  # Ganti dengan path video default Anda
        if os.path.exists(default_video_path):
            uploaded_video = open(default_video_path, 'rb')
            default_file_name = os.path.basename(default_video_path)
        else:
            st.stop()
    else:
        default_file_name = uploaded_video.name

    # Save uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # Display the name of the uploaded or default video
    st.write(f"File: {default_file_name}")

    # Extract frames from the video
    video_capture = cv2.VideoCapture("temp_video.mp4")
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frame_rate = 10  # Extract 10 frames per second
    frame_interval = int(fps / frame_rate)

    frames = []
    frame_times = []
    count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
            frame_times.append(count / fps)
        count += 1

    video_capture.release()

# Initialize detection results
detection_results = []

# Folder to save frames
video_name = default_file_name.split('.')[0]
folder_name = f"{video_name}_frames"
os.makedirs(folder_name, exist_ok=True)

# Tab 2: Preview Frame
with tab2:
    # Display extracted frames section title and download button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Frame Video Extract")
        # Add real-time frame counter
        frame_counter_placeholder = st.empty()
    with col2:
        # Create a ZIP file containing all frames in memory
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, 'w') as zip:
            for i in range(len(frames)):
                frame_path = f"{folder_name}/detected_frame_{i+1}.jpg"
                if os.path.exists(frame_path):
                    zip.write(frame_path, arcname=f"detected_frame_{i+1}.jpg")
        zip_buffer.seek(0)

        # Display the download button for the ZIP file
        st.download_button(
            label="Download Zip Frame",
            data=zip_buffer,
            file_name=f"{folder_name}.zip",
            mime="application/zip"
        )

    # Process frames and display them in a grid with scrollable and expander
    cols_per_row = 10
    with st.expander("View Frames"):
        container = st.container(height=350)
        for idx, frame in enumerate(frames, start=1):
            start_time = datetime.now()
            img_with_detections, detected_faces = detect_and_predict_emotion(frame)
            end_time = datetime.now()
            detection_time = (end_time - start_time).total_seconds()
            
            # Update the frame counter
            frame_counter_placeholder.text(f"Processing frame {idx}/{len(frames)}")
            
            # Count emotions
            emotion_counts = {emotion: 0 for emotion in emotions}
            for (_, _, _, _, emotion) in detected_faces:
                emotion_counts[emotion] += 1
            
            total_emotions = sum(emotion_counts.values())
            
            emotion_percentages = {f'{emotion} (%)': (count / total_emotions * 100 if total_emotions > 0 else 0) for emotion, count in emotion_counts.items()}
            
            detection_results.append({
                'Frame': str(idx),  # Ubah tipe data Frame menjadi string
                'Time (s)': frame_times[idx-1],
                'Detection Time (s)': detection_time,
                'N': total_emotions,
                **emotion_counts,
                **emotion_percentages
            })

            # Save detected frame with annotations directly to the folder
            normalized_filename = f"{folder_name}/detected_frame_{idx}.jpg"
            cv2.imwrite(normalized_filename, img_with_detections)

            # Resize the frame for display in Streamlit
            resized_img_with_detections = cv2.resize(img_with_detections, (150, 100))

            # Display the image in the correct column
            if (idx - 1) % cols_per_row == 0:
                cols = container.columns(cols_per_row)
            
            col_idx = (idx - 1) % cols_per_row
            with cols[col_idx]:
                st.image(cv2.cvtColor(resized_img_with_detections, cv2.COLOR_BGR2RGB), caption=f"Frame {idx}", use_column_width=True)

# Create a DataFrame from detection results
df_results = pd.DataFrame(detection_results)

# Fill any NaN or empty strings with 0
df_results.fillna(0, inplace=True)

# Add totals row
totals = df_results[emotions].sum().to_dict()
totals_percentages = {f'{emotion} (%)': (totals[emotion] / sum(totals.values()) * 100 if sum(totals.values()) > 0 else 0) for emotion in emotions}
totals['Frame'] = 'Total'
totals['Time (s)'] = 0.0
totals['Detection Time (s)'] = df_results['Detection Time (s)'].sum()
totals['N'] = df_results['N'].sum()
totals.update(totals_percentages)
totals_df = pd.DataFrame([totals])
df_results = pd.concat([df_results, totals_df], ignore_index=True)

# Tab 3: Export Data
with tab3:
    # Apply custom CSS to limit the width of the table
    st.markdown(
        """
        <style>
        .dataframe-container {
            width: 700px;  /* Adjust the width as needed */
            margin: 0 auto;
        }
        .download-button {
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the dataframe with emotion counts and download button
    col1, col2 = st.columns([4, 1])  # Adjusted column widths to make col1 wider
    with col1:
        st.markdown("### Emotion Counts Data")
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        # Download button for the emotion counts data as CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.markdown('<div class="download-button">', unsafe_allow_html=True)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{video_name}_emotion_counts.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    st.write(df_results)  # Use st.write to display the DataFrame

    st.markdown("<h1 style='text-align: center; color: white; font-size: 24px;'>Emotion Distribution Across All Frames</h1>", unsafe_allow_html=True)

    # Plot pie chart
    pie_colors = ['#F7464A', '#33CF49', '#4684F7', '#F7B246']
    plt.figure(figsize=(6, 6))  # Adjust figure size as needed
    plt.pie(df_results[emotions].sum(), labels=emotions, autopct='%1.1f%%', startangle=140, colors=pie_colors, textprops={'color': 'white'}, wedgeprops={'edgecolor': 'black'})

    # Create columns for layout
    col1, col2 = st.columns([3, 2])  # Adjust column widths as needed

    # Display pie chart in the right column
    with col1:
        st.pyplot(plt, transparent=True)

    # Display explanation in the left column
    with col2:
        st.write("Explanation of Pie Chart Colors")
        st.markdown("""
        <span style="color: #F7464A;">&#9679;</span> **Anger**<br>
        <span style="color: #33CF49;">&#9679;</span> **Happy**<br>
        <span style="color: #4684F7;">&#9679;</span> **Sad**<br>
        <span style="color: #F7B246;">&#9679;</span> **Neutral**<br>
        """, unsafe_allow_html=True)
        
    st.title("Hasil Deteksi")

    # Calculate percentages
    emotion_sums = df_results[emotions].sum()
    total_counts = emotion_sums.sum()
    percentages = {emotion: (count / total_counts) * 100 for emotion, count in emotion_sums.items()}
    
    # Sort percentages
    sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

    # Create the markdown string
    markdown_str = ', '.join([f"{value:.1f}% {key}" for key, value in sorted_percentages])

    # Display the markdown string
    with st.container():
        st.markdown(markdown_str)
