# PACKAGES
import streamlit as st
import cv2
import datetime
import numpy as np
import tempfile
import os
from objTracker import * # IMPORTANT: objTracker.py must be in the same directory

# --- STREAMLIT APP CONFIGURATION ---
st.set_page_config(page_title="Vehicle Speed Tracker", layout="wide")
st.title("🚗 Vehicle Speed Tracker")
st.info("Upload a traffic video, adjust the tuning parameters in the sidebar, then click 'Start Processing'.")

# --- SIDEBAR FOR TUNING ---
st.sidebar.header("⚙️ Tuning Parameters")
st.sidebar.info(
    "Adjust these values if no vehicles are being detected. "
    "Every video requires different settings."
)

# --- NEW: DEBUG CHECKBOX ---
debug_mask = st.sidebar.checkbox("Show Debug Mask")
st.sidebar.caption(
    "Check this box to see the black & white mask. "
    "Your goal is to make vehicles appear as solid white shapes."
)

thresh_value = st.sidebar.slider("1. Binary Threshold Value", 0, 255, 200)
st.sidebar.caption(
    "Lower this value if vehicles are not appearing on the debug mask. "
    "Try values between 50-150."
)

area_value = st.sidebar.slider("2. Minimum Contour Area", 100, 5000, 1000)
st.sidebar.caption(
    "Increase this to filter out small noise. "
    "Decrease it if small vehicles are being missed."
)


# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # --- START PROCESSING BUTTON ---
    if st.button("Start Processing"):
        
        # Placeholder for the video feed
        st_frame = st.empty()
        # Placeholder for results
        st_results_title = st.empty()
        st_results_data = st.empty()
        st_results_plot_info = st.empty()
        st_results_plot = st.empty()

        # --- ORIGINAL SCRIPT LOGIC ---
        
        # TRACKER OBJ
        tracker = EuclideanDistTracker()

        # CAPTURE INPUT VIDEO STREAM
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Ensure total_frames is not zero to avoid division by zero
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                st.error("Error: Could not get video frame count. Is the video file valid?")
                cap.release()
                if os.path.exists(video_path):
                    os.remove(video_path)
                st.stop()
                
            frame_count = 0

            # KERNALS
            kernalOp = np.ones((3, 3), np.uint8)
            kernalCl = np.ones((11, 11), np.uint8)
            # kernalEr = np.ones((5, 5), np.uint8) # This was removed
            fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

            # Progress bar
            progress_bar = st.progress(0, text="Processing video...")

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break  # End of video

                frame_count += 1

                # --- Processing Logic ---
                try:
                    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                except cv2.error as e:
                    st.warning(f"Could not resize frame {frame_count}. Skipping. Error: {e}")
                    continue
                    
                height, width, _ = frame.shape
                roi = frame[50:540, 200:960]

                # MASKING
                fgmask = fgbg.apply(roi)
                
                # --- USE SLIDER VALUE FOR THRESHOLD ---
                ret, binImg = cv2.threshold(fgmask, thresh_value, 255, cv2.THRESH_BINARY)
                
                opening = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernalOp)
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernalCl)
                
                # --- MODIFIED: Removed erode step ---
                # We find contours on the 'closing' image, which is more robust
                processed_mask = closing

                # CONTOURS & BOUNDING BOX
                contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                detections = []

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    
                    # --- USE SLIDER VALUE FOR AREA ---
                    if area > area_value:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Draw detection rectangle on the original 'roi'
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        detections.append([x, y, w, h])

                # OBJ TRACKING
                boxes_ids = tracker.update(detections)
                for box_id in boxes_ids:
                    x, y, w, h, id = box_id

                    try:
                        s = tracker.getsp(id)
                        if (s < tracker.limit()):
                            cv2.putText(roi, str(id) + " " + str(s), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        else:
                            cv2.putText(roi, str(id) + " " + str(s), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)

                        if (tracker.f[id] == 1 and s != 0):
                            tracker.capture(roi, x, y, h, w, s, id)
                    except Exception as e:
                        st.warning(f"Error processing object ID {id}: {e}")


                # DRAW LINES
                cv2.line(roi, (0, 410), (960, 410), (255, 0, 0), 2)
                cv2.putText(roi, 'START', (2, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.line(roi, (0, 430), (960, 430), (255, 0, 0), 2)
                cv2.line(roi, (0, 235), (960, 235), (255, 0, 0), 2)
                cv2.putText(roi, 'END', (2, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.line(roi, (0, 255), (960, 255), (255, 0, 0), 2)

                # DISPLAY DATE, TIME, FPS & CURRENT FRAME
                cv2.line(roi, (0, 10), (960, 10), (79, 79, 47), 30)
                d = str(datetime.datetime.now().strftime("%d-%m-%y"))
                t = str(datetime.datetime.now().strftime("%H-%M-%S"))
                cv2.putText(roi, f'DATE: {d} |', (25, 19), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
                cv2.putText(roi, f'TIME: {t} |', (209, 19), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
                cv2.putText(roi, f'FPS: {fps:.2f} |', (393, 19), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
                cv2.putText(roi, f'FRAMES: {frame_count} of {total_frames} ', (510, 19), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
                cv2.line(roi, (0, 26), (960, 26), (255, 255, 255), 2)
                
                # --- STREAMLIT DISPLAY ---
                if debug_mask:
                    # Show the black & white mask to help with tuning
                    st_frame.image(processed_mask, use_column_width=True)
                else:
                    # Show the final video with bounding boxes
                    st_frame.image(roi, channels="BGR", use_column_width=True)
                
                # Update progress bar
                progress_bar.progress(frame_count / total_frames, text=f"Processing video... ({frame_count}/{total_frames})")

            # --- AFTER THE LOOP ---
            progress_bar.progress(1.0, text="Processing complete!")
            st.success("Video processing complete!")
            
            # Clean up
            cap.release()
            if os.path.exists(video_path):
                os.remove(video_path) # Delete the temporary file

            # Final tracker calls
            tracker.end() # This just passes, but we call it to be safe
            ids_lst, spd_lst = tracker.dataset()

            # --- DISPLAY RESULTS ---
            st_results_title.subheader("📊 Processing Results")
            if ids_lst and spd_lst:
                st_results_data.dataframe({"Vehicle ID": ids_lst, "Speed (px/s)": spd_lst})
                
                # --- Display the plot ---
                st_results_plot_info.info("Note: Speed is calculated in pixels/second (px/s) based on the detection lines.")
                try:
                    fig = tracker.datavis(ids_lst, spd_lst)
                    if fig:
                        st_results_plot.pyplot(fig)
                    else:
                        st_results_plot.warning("`datavis` did not return a figure object.")
                except Exception as e:
                    st_results_plot.error(f"Error calling datavis: {e}")
            else:
                st_results_data.warning("No vehicles were tracked successfully.")
