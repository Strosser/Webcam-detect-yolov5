import cv2
import torch
import numpy as np

#path to model
model_path = 'yolov5/yolov5s.pt'
# Load existing model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)


# Function to detect objects
def detect_objects(frame):
    results = model(frame)
    return results


# Using webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
    results = detect_objects(frame)

    # Getting results
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

    # Going through every detected object
    for i in range(len(labels)):
        row = cord[i]
        if row[4] >= 0.25:  # Probability
            x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(
                row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
            label = model.names[int(labels[i])]
            confidence = row[4]
            # Label and shape
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
                        2)

    # Zobrazení výsledků
    cv2.imshow('YOLOv5 Detection', frame)

    # Ending with keycap 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
