from ultralytics import YOLO

# Training downloaded Roboflow Dataset
path = r'D:\Internship\dataextraction\FFB counting.v1i.yolov8\data.yaml'
trainingModel = YOLO('yolov8n.yaml')  # creates a new model

results = trainingModel.train(data=path, epochs=100)