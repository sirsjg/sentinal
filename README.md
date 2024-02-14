# Sentinal

A fun example of using OpenCV for real-time object detection and could be used as a basis for various applications, such as security systems or automated monitoring tools.  This will detect people, cats, dogs and cars.

CV will detect if someone is close by for more than a few seconds, then log the activity as well as record them.

Press '***q***' to quit.

## Dependencies

```bash
 pip install -r requirements.txt
 ```

Also will need the YOLO4 pre-trained model. [YOLO4 weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and [YOLO4 config](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg).
