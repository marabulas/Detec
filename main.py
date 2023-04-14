import random
import time

import torch
import cv2
from pathlib import Path

from compare_image import compare
from upscale import upscale, upscale_in_background

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).cuda().eval()

# Path to the video to be detected
video_path = Path("video.mp4")

# Load the video using OpenCV
cap = cv2.VideoCapture(str(video_path))

# Get the video frames per second and size
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Create a VideoWriter object to save the video with the detections
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, size)

# Initialize variables for tracking people
people = []
px_to_m = 0.03
min_distance_threshold = 20

# Loop through the frames of the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Stop the loop if the end of the video is reached
    if not ret:
        break

    # Run the model on the frame
    results = model(frame, size=640)

    # Loop through the detected objects
    for i, det in enumerate(results.xyxy[0]):
        # Extract the class name, confidence score, and bounding box coordinates
        class_name = results.names[int(det[5])]
        confidence = det[4]
        bbox = [int(x) for x in det[:4]]

        # Extract the portion of the image where the object was detected
        x1, y1, x2, y2 = bbox

        //TODO PASSAR TODOS OS DADOS DE TUDO DETECTADO, Nâo APENAS PESSOAS
        //MODIFICAR PESSOAS PARA NÃO ENCHER A ARRAY



        if class_name == "person":

            object_img = frame[y1:y2, x1:x2]


            # Check if this person was detected before
            detected = False
            for person in people:
                # Calculate the x and y difference between the current frame and the old frame
                delta_x = bbox[0] - person['x']
                delta_y = bbox[2] - person['y']
                distance = ((delta_x) ** 2 + (delta_y) ** 2) ** 0.5
                if distance < min_distance_threshold:
                    # Update the person's data
                    person['x'] = bbox[0]
                    person['y'] = bbox[2]
                    person['last_detected'] = time.time()
                    person['speed'] = 0
                    person['detected'] = True
                    person['x_history'].append(bbox[0])
                    person['y_history'].append(bbox[2])
                    person['frames_count'] += 1
                    detected = True
                    break

            if not detected:
                r = random.randint(75, 255)
                g = random.randint(75, 255)
                b = random.randint(75, 255)
                color = (r, g, b)
                # Add the person's data to the people list
                person = {
                    'id': i,
                    'x': bbox[0],
                    'y': bbox[2],
                    'last_detected': time.time(),
                    'speed': 0,
                    'detected': True,
                    'color': color,
                    'x_history': [bbox[0]],
                    'y_history': [bbox[2]],
                    'frames_count': 0,
                    'speeds': [0],
                    'snapshot': object_img
                }
                people.append(person)



            # Draw a rectangle around the object
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), person['color'], 2)

            # Put the object's label and confidence score on the rectangle
            label = f"Person: {person['id']} - c: {class_name} {confidence:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)




    # Remove the people that were not detected in this frame
    people = [person for person in people if person['detected']]
    print(len(people))



    # Display the results on the frame
    #Must call this and remove ALL CV2 putSomething to render the AI preview
    #results.render()

    # Save the frame with the detections
    out.write(frame)

    # Display the frame with the detections
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and writer objects, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()




