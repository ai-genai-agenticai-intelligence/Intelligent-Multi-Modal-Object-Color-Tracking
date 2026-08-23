#it is a track it customer detect and vechicle in the vedio and count the nunber of customer and vechicle in the vedio
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model trained for customer detection
model = YOLO('yolov8n.pt')  # Replace with the path to your trained model if necessary")

import os

# Initialize the video capture object (falls back to webcam 0 if sample video is not found)
video_path = r'vedio_sample3.mp4'
cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Customer Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()