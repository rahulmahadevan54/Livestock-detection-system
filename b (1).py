import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO(r'D:\python (1)\runs (1)\detect (1)\train3\weights\best.pt')

# Video path
video_path = r'D:\r.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize previous frame for motion detection
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Convert the first frame to grayscale for motion detection
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Idle status buffer
idle_buffer = []
buffer_size = 5  # Number of frames to consider for idle detection

# List to store positions of detected cows
cow_positions = {}

# Loop
while True:
    # Read a frame
    success, frame = cap.read()
    if not success:
        print("End of video.")
        break

    # Convert current frame to grayscale
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and the previous frame
    motion = cv2.absdiff(prev_frame, current_frame)

    # Threshold the motion to highlight significant changes
    _, motion_threshold = cv2.threshold(motion, 30, 255, cv2.THRESH_BINARY)

    # Detect cows in the current frame using YOLO
    results = model(frame)
    boxes = results[0].boxes.xyxy.numpy()
    confidences = results[0].boxes.conf.numpy()

    # Convert boxes and confidences to required format for NMS
    boxes_list = [tuple(map(int, box)) for box in boxes]
    confidences_list = confidences.tolist()

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.8, 0.3)

    # Initialize cow count and idle/active status
    cow_count = 0
    idle_count = 0
    active_count = 0

    # Process detections after applying NMS, if there are any
    if len(indices) > 0:  # Ensure indices are not empty
        for i in indices:
            index = i[0] if isinstance(i, (list, np.ndarray)) else i  # Extract the index safely
            confidence = confidences[index]
            x1, y1, x2, y2 = map(int, boxes[index])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cow_count += 1  # Increment total cow count

            # Track the cow's position
            cow_id = f'cow_{index}'  # Assign a unique ID to each cow

            # Check for motion to determine idle status
            if cow_id not in cow_positions:
                cow_positions[cow_id] = [(x1, y1, x2, y2)]  # Store the initial position
                idle_buffer.append(1)  # Mark as idle for the first frame
            else:
                # Check if position changed significantly
                previous_position = cow_positions[cow_id][-1]
                position_changed = (abs(previous_position[0] - x1) > 5 or
                                    abs(previous_position[1] - y1) > 5 or
                                    abs(previous_position[2] - x2) > 5 or
                                    abs(previous_position[3] - y2) > 5)
                
                if not position_changed:  # No significant motion detected
                    idle_buffer.append(1)  # Mark as idle
                else:
                    idle_buffer.append(0)  # Mark as active

                cow_positions[cow_id].append((x1, y1, x2, y2))  # Update position history

            # Count idle/active cows based on the buffer
            if len(idle_buffer) > buffer_size:
                idle_buffer.pop(0)  # Maintain buffer size

            if sum(idle_buffer) == len(idle_buffer):  # All frames in buffer are idle
                idle_count += 1
            else:
                active_count += 1

            # Display confidence level
            cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display cow counts on the frame
    cv2.putText(frame, f'Total Cows: {cow_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Active Cows: {active_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Idle Cows: {idle_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Normalize the heatmap for visualization
    heatmap_normalized = cv2.normalize(motion_threshold.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay heatmap on the original frame
    overlay = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)

    # Display the overlayed frame with heatmap
    cv2.imshow("Cow Detection and Motion Heatmap", overlay)

    # Update previous frame
    prev_frame = current_frame

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
