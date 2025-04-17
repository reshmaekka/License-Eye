# License Plate Detection and Recognition System

This project uses YOLOv8 and EasyOCR to detect and recognize license plates from a webcam feed and verify them against a preloaded car owner database.

## 🚀 Features

- Real-time license plate detection using YOLOv8.
- Optical character recognition (OCR) with EasyOCR to extract plate numbers.
- Cross-check license plates with a CSV-based database.
- Displays owner info if the plate is registered.
- Color-coded feedback for registered/unregistered plates.

## 🛠️ Requirements

Install the following dependencies before running the project:

```bash
pip install opencv-python numpy pandas easyocr ultralytics
