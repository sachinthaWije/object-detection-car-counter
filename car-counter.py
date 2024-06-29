import cv2
from ultralytics import YOLO

import cvzone
import math

from sort import *

cap = cv2.VideoCapture('your video path')
# https://www.youtube.com/watch?v=G3shG3msavM&t=11s
# I used this video


# Load the YOLO model
model = YOLO('../YoloModel/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load the mask image (optional, can be used to define regions of interest)
mask = cv2.imread("your mask image")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


# Define the up and down lines for counting vehicles
limits_up = [123, 497, 600, 497]
limits_down = [700, 597, 1290, 597]

# Lists to hold the unique IDs of counted vehicles
totalCount_up = []
totalCount_down = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    result = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if ((currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus')
                    and conf > 0.3):
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 100), 5)
    cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 0, 100), 5)

    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (100, 0, 100), cv2.FILLED)

        # Check if the vehicle up
        if limits_up[0] < cx < limits_up[2] and limits_up[1] - 15 < cy < limits_up[1] + 15:
            if totalCount_up.count(Id) == 0:
                totalCount_up.append(Id)
                cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 180, 0), 5)

        # Check if the vehicle down
        if limits_down[0] < cx < limits_down[2] and limits_down[1] - 15 < cy < limits_down[1] + 15:
            if totalCount_down.count(Id) == 0:
                totalCount_down.append(Id)
                cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 180, 0), 5)

    cvzone.putTextRect(img, f' Count Up: {len(totalCount_up)}', (50, 50), colorR=(50, 100, 50))
    cvzone.putTextRect(img, f' Count Down: {len(totalCount_down)}', (850, 50), colorR=(50, 100, 50))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
