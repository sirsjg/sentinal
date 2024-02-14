import cv2
import numpy as np
import time
import json

# Load YOLO model and class labels
model_config = './yolov4.cfg'
model_weights = './yolov4.weights'
class_labels = './coco.names'

# Path to the JSON file
json_file = 'log.json'

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
with open(class_labels, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Set the minimum confidence level for detections
confidence_threshold = 0.5

# Set the desired classes to detect
desired_classes = ['person', 'car', 'cat', 'dog']

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Initialize person detection variables
person_present = False
person_start_time = None

# Initialize notification variables
notification_sent = False
notification_start_time = None

# Main loop
while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Run YOLO object detection on the frame
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [(layer_names[i[0] - 1])
                     for i in np.reshape(net.getUnconnectedOutLayers(), (3, 1))]
    detections = net.forward(output_layers)

    # Process the detections
    boxes = []
    confidences = []
    class_ids = []
    person_detected = False

    for detection in detections:
        for detection_result in detection:
            scores = detection_result[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold and labels[class_id] in desired_classes:
                if labels[class_id] == 'person':
                    person_detected = True
                    if not person_present:
                        person_present = True
                        person_start_time = time.time()
                        notification_sent = False
                        notification_start_time = None
                else:
                    person_present = False
                    person_start_time = None

                center_x = int(detection_result[0] * frame.shape[1])
                center_y = int(detection_result[1] * frame.shape[0])
                width = int(detection_result[2] * frame.shape[1])
                height = int(detection_result[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if person_detected and person_present and time.time() - person_start_time > 3:
        if not notification_sent:
            # Notify when a person has been seen for 3 seconds or more
            print("Person detected for 3 seconds or more")
            notification_sent = True
            notification_start_time = time.time()

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            size = (frame_width, frame_height)
            writer = cv2.VideoWriter(
                str(notification_start_time) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        else:
            writer.write(frame)

    elif not person_detected and notification_sent and time.time() - notification_start_time > 3:
        # New object to add
        new_object = {
            'event': 'person_detected',
            'start': notification_start_time,
            'finish': time.time()
        }

        # Load existing JSON data from the file
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Append the new object to the array
        json_data.append(new_object)

        # Write the updated JSON data back to the file
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)

        # Restart the counter if the person moves away for more than 3 seconds
        person_present = False
        person_start_time = None
        notification_sent = False
        notification_start_time = None
        writer.release()

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw bounding boxes and labels
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        label = f'{labels[class_ids[i]]}: {confidences[i]:.2f}'

        cv2.rectangle(frame, (left, top), (left + width,
                      top + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with overlays
    cv2.imshow('Webcam Stream', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
