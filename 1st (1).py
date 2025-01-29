import cv2
import os

#Define the path to the video file and the output directory
video_path = r'C:\Users\rahul\Downloads\cow.mp4'
output_dir = r'C:\Users\rahul\Documents\frames' 

#Create the output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Open the video file
cap = cv2.VideoCapture(video_path)

#Check if video opened
if not cap.isOpened():
    print("Error: Error in video.")
else:
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % 30 == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frame_count += 1
            frame_count += 1
        else:
            break

    cap.release()

print(f"Number of frames saved: {saved_frame_count}")