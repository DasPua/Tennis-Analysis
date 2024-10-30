from ultralytics import YOLO

model = YOLO('models\\yolov5xu_best.pt')

image_path = "C:\\Users\\91637\\.vscode\\tennis_pt\\input_files\\test_vid.mp4"

result=model.predict(image_path, conf= 0.2,save=True)

print("bounding boxes:")

for box in result[0].boxes:
    print(box)