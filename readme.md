# Face Mask Detection System

This project is a Face Mask Detection System that uses a TensorFlow model to detect whether a person is wearing a mask or not. The system can process input from a webcam, a saved video file, or an image file.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aayush-ojha/face-mask-detection-system.git
    cd face-mask-detection-system
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Model

The TensorFlow model used in this project can be downloaded from [AIZOOTech/FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection/blob/master/models/face_mask_detection.pb). Ensure you have the `face_mask_detection.pb` file in the project directory.

## Usage

1. Ensure you have the TensorFlow model file (`face_mask_detection.pb`) in the project directory.

2. Run the script with the desired input source:

    - To process input from the webcam:
        ```python
        python3 main.py
        ```

    - To process a saved video file:
        ```python
        python3 main.py --video path_to_video.mp4
        ```

    - To process an image file:
        ```python
        python3 main.py --image path_to_image.jpg
        ```

## Code Overview

- [load_model(model_path)](http://_vscodecontentref_/0): Loads the TensorFlow model from the specified path.
- [detect_mask(sess, graph, frame, face_cascade, input_tensor, output_tensor)](http://_vscodecontentref_/1): Detects faces in the frame and determines if they are wearing masks.
- `process_webcam(graph, face_cascade, input_tensor, output_tensor)`: Captures video from the webcam, processes each frame, and saves the output to `output_webcam.avi`.
- [process_video(graph, face_cascade, input_tensor, output_tensor, video_path)](http://_vscodecontentref_/2): Processes a saved video file and saves the output to [output_video.avi](http://_vscodecontentref_/3).
- `process_image(graph, face_cascade, input_tensor, output_tensor, image_path)`: Processes a saved image file and saves the output to `output_image.jpg`.