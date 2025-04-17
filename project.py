import cv2
import numpy as np
import YOLO
from ultralytics import YOLO
import easyocr
import pandas as pd

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Use 'en' for English

# Load the pre-trained YOLOv8 model for object detection
model = YOLO('yolov8n.pt')  # YOLOv8 Nano Model

# Load car database (CSV format)
def load_database(csv_file):
    try:
        return pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return pd.DataFrame()

# Check if the license plate is registered
def check_license_plate(database, plate_number):
    entry = database[database['LicensePlate'] == plate_number]
    if not entry.empty:
        return True, entry.iloc[0]['OwnerName']
    return False, None

# Function to extract text using EasyOCR
def extract_text_from_image(image):
    result = reader.readtext(image)
    if result:
        # Extract the first detected text as the license plate
        return result[0][1]  # EasyOCR returns a list of tuples (bbox, text, confidence)
    return ""

def main():
    # Load car owner dataset (CSV file)
    database = load_database('car_database.csv')

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit the program")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 model on the frame to detect license plates
        results = model(frame)
        for result in results:
            boxes = result.boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)
            confidences = result.boxes.conf  # Confidence scores
            class_ids = result.boxes.cls  # Class IDs

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Extract coordinates and class ID
                x1, y1, x2, y2 = map(int, box)
                cropped_plate = frame[y1:y2, x1:x2]

                # Extract text (license plate number) using EasyOCR
                plate_number = extract_text_from_image(cropped_plate)

                if plate_number:
                    # Check if the license plate is registered
                    registered, owner_name = check_license_plate(database, plate_number)

                    # Display license plate info on the frame
                    label = f"Plate: {plate_number}"
                    if registered:
                        label += f" - Registered to {owner_name}"
                        color = (0, 255, 0)  # Green for registered
                    else:
                        label += " - Unregistered"
                        color = (0, 0, 255)  # Red for unregistered

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the processed frame
        cv2.imshow("Webcam License Plate Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
