from ultralytics import YOLO


# Path to the trained model
MODEL_PATH = "best.pt"

# Path to the input video
VIDEO_PATH = "input_video.mp4"


# Load the trained helmet detection model
model = YOLO(MODEL_PATH)

print("Trained helmet detection model loaded successfully!")


# Run helmet detection on the video
results = model.predict(
    source=VIDEO_PATH,
    conf=0.25,
    save=True,
    project="runs",
    name="helmet_video_prediction"
)

print("Video processing completed successfully!")
print("The annotated video has been saved in the runs directory.")
