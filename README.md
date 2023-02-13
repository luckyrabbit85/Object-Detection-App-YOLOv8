# Object Detection App with YOLOv8 and Gradio

A real-time object detection app built using YOLOv8 and Gradio.

## Installation and Usage

Create a a Virtual Environment with Python 3.10 and install the required dependencies, then run the script:

```
    pip install -r requirements.txt
    python app.py
```

## Description

This app utilizes YOLOv8, a state-of-the-art object detection model, to identify objects within an image in real-time. The user can input an image and the app will display the bounding boxes and labels for the detected objects. The app is built using Gradio, a platform for building and sharing interactive machine learning models, which allows for easy integration and deployment of the YOLOv8 model.

## Requirements

The following libraries are required to run the app:

- ultralytics
- gradio
- rich
- gdown

## Credits

- YOLOv8: [https://github.com/ultralytics/yolov8](https://github.com/ultralytics/yolov8)
- Gradio: [https://gradio.app/](https://gradio.app/)
