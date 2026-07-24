from ultralytics import YOLO


# Load pretrained YOLO11 Nano model
model = YOLO("yolo11n.pt")


# Train the model
results = model.train(
    data="dataset/new dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    device=0,
    project="runs",
    name="helmet_detection_test"
)


# Validate the trained model
metrics = model.val()

print("Training and validation completed successfully!")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
