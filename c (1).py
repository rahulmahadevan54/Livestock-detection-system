import cv2
from ultralytics import YOLO

#Load the model
model = YOLO(r'C:\Users\rahul\python\runs\detect\train2\weights\best.pt')

#Open the video file
video_path = r'C:\Users\rahul\Downloads\m.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
#Loop
while cap.isOpened():
    #Read a frame
    success, frame = cap.read()

    if success:
        #Run YOLOv8 on frame
        results = model(frame)

        #Visualize the results
        annotated_frame = results[0].plot()

        #Display the annotated frame
        cv2.imshow("YOLOv8", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
