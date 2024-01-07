from collections import defaultdict

import cv2
import numpy as np
import time
from ultralytics import YOLO
import time
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path =0
cap = cv2.VideoCapture(video_path)

'''
write your code here for the task'''

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # Get the boxes and track IDs
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            boxes1= results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            for i in range(8):
                print("No Detection detected in th frame")
                k=open("ko.txt","w")
            continue
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        f=open("trackdf.txt","w")
        f.write('[\n')
        for box, track_id in zip(boxes, track_ids):
            last={"X":float(box[0]),"Y":float(box[1]),"W":float(box[2]),"H":float(box[3]),"id":track_id}
            f.write(str(last))
            f.write(",\n")
            cv2.circle(annotated_frame, (int(box[0]),int(box[1])), 2, (255,255,255), 2)
        f.write(']\n')
        f.write("[\n")
        for box, track_id in zip(boxes1, track_ids):
            last={"x1":float(box[0]),"y1":float(box[1]),"x1":float(box[2]),"y2":float(box[3]),"id":track_id}
            f.write(str(last))
            f.write(",\n")
            cv2.circle(annotated_frame, (int(box[0]),int(box[1])), 2, (255,255,255), 2)
            cv2.circle(annotated_frame, (int(box[2]),int(box[3])), 2, (255,255,255), 2)
            cv2.circle(annotated_frame, (int(box[0]),int(box[3])), 2, (255,255,255), 2)
            cv2.circle(annotated_frame, (int(box[2]),int(box[1])), 2, (255,255,255), 2)
        f.write("]")
        f.close()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1)==ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break


# Release the video capture object and close the display window
cap.release
cv2.destroyAllWindows()