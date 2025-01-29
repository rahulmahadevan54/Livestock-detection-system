from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt') 

# Train the model
results = model.train(data=r'C:\Users\rahul\Documents\json_dir\YOLODataset\dataset.yaml', epochs=100, imgsz=640)