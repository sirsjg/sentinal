import cv2
import numpy as np
import requests

# Load YOLO model
model_config = 'path/to/yolov3.cfg'
model_weights = 'path/to/yolov3.weights'
class_labels = 'path/to/coco.names'

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
with open(class_labels, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Set the minimum confidence level for detections
confidence_threshold = 0.5

# Set the desired classes to detect
# Add or remove classes as needed
desired_classes = ['person', 'car', 'cat', 'dog']

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set the output video size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the HTTP endpoint to stream the video
stream_url = 'http://localhost:8080/stream.mjpg'

# Create an MJPEG stream writer
stream_writer = cv2.VideoWriter(stream_url, cv2.VideoWriter_fourcc(
    *'MJPG'), 10, (frame_width, frame_height))

# Main loop
while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Run YOLO object detection on the frame
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    # Process the detections
    boxes = []
    confidences = []
    class_ids = []
    for detection in detections:
        for detection_result in detection:
            scores = detection_result[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold and labels[class_id] in desired_classes:
                center_x = int(detection_result[0] * frame_width)
                center_y = int(detection_result[1] * frame_height)
                width = int(detection_result[2] * frame_width)
                height = int(detection_result[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw bounding boxes and labels
    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box
        label = f'{labels[class_ids[i]]}: {confidences[i]:.2f}'

        cv2.rectangle(frame, (left, top), (left + width,
                      top + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with overlays
    cv2.imshow('Webcam Stream', frame)

    # Write the frame to the stream writer
    stream_writer.write(frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
stream_writer.release()
cv2.destroyAllWindows()
