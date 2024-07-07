# YOLOv5 Webcam Object Detection

This project demonstrates real-time object detection using YOLOv5 and a webcam. The model detects objects in video streams captured from a webcam and displays the results with bounding boxes and labels.

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- Python 3.7 or later
- `git` installed on your system

### Clone the Repository

```sh
git clone https://github.com/Strosser/Webcam-detect-yolov5.git
cd Webcam-detect-yolov5
```

### Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```sh
# Create a virtual environment
python3 -m venv yolo_env

# Activate the virtual environment
source yolo_env/bin/activate
```

### Install Dependencies

Install the necessary packages using `pip`.

```sh
pip install -r requirements.txt
```

### Download the YOLOv5 Model

Download the pre-trained YOLOv5 model.

```sh
git clone https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```

## Usage

To run the object detection script, use the following command:

```sh
python detect_webcam.py
```

This script will start your webcam and display a window with detected objects highlighted by bounding boxes and labels.

### Stopping the Script

To stop the script, focus on the display window and press the 'q' key.

## Configuration

You can configure various parameters in the script to suit your needs:

- **Model Path**: Change the path to the YOLOv5 model if necessary.
- **Confidence Threshold**: Adjust the confidence threshold to filter detections.
- **Webcam Source**: Change the webcam source if you have multiple webcams.

## How It Works

1. **Model Loading**: The YOLOv5 model is loaded using the PyTorch Hub.
2. **Webcam Initialization**: The script initializes the webcam using OpenCV.
3. **Frame Capture and Processing**: Frames are captured from the webcam and passed to the YOLOv5 model for object detection.
4. **Result Visualization**: Detected objects are highlighted with bounding boxes and labels on the video stream.
5. **Display**: The processed video stream is displayed in a window.

### Key Components

- **YOLOv5 Model**: Pre-trained model for object detection.
- **OpenCV**: Library for real-time computer vision.
- **PyTorch**: Deep learning framework.
