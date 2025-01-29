import cv2
from ultralytics import YOLO

# Load the custom model
model = YOLO(r'C:\Users\rahul\python\runs\detect\train3\weights\best.pt')

video_path = r'C:\Users\rahul\Downloads\cow.mp4'
cap = cv2.VideoCapture(video_path)

# Define the line coordinates for the counting line
START = (0, 700)
END = (1400, 700)

# Store the track history and crossed cattle
track_history = {}
crossed_objects = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    results = model(frame)

    # Check if results contain boxes
    if hasattr(results, 'xyxy') and results.xyxy is not None:
        boxes = results.xyxy[0].cpu().numpy()
        class_ids = results.xyxy[1].cpu().numpy()

        # Process each detected object
        for box, class_id in zip(boxes, class_ids):
            x, y, w, h = box
            center_x, center_y = int(x + w / 2), int(y + h / 2)

            # Update the track history
            if class_id not in track_history:
                track_history[class_id] = []
            track_history[class_id].append((center_x, center_y))

            # Check if the object crosses the line
            if START[0] < center_x < END[0] :
                crossed_objects.add(class_id)

    # Visualize the results on the frame
    annotated_frame = results.render()[0]

    # Draw the line on the frame
    cv2.line(annotated_frame, START, END, (255, 0, 0), 2)

    # Write the count of objects on each frame
    count_text = f"Cattle counter: {len(crossed_objects)}"
    cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YOLOv8 - cattle counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()