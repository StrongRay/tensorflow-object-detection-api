#
#   Program :  O3.py
#
#   Description:  Loads detection model and label based on ssd_mobilenet_v1_coco and use webcam to detect 
#
#   Reference:  https://github.com/tensorflow/models/tree/master/research/object_detection
#               https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
#
#   Environment: UBUNTU 18.04, tensorflow 1.12, OpenCV 4.0.0-alpha
#
#   Features:  
#
#   1.  Remove expensive cv2.waitKey() and replace with plt
#   2.  Some debug functions to print duration of each detection - duration("start","xxx") and duration("end","xxx")
#
#   Improvement
#
#   a.  Graceful exit of program [nice to have]
#   b.  Accept arguments for specifying camera source [nice to have] 
#   c.  Hide toolbar of plt [cosmetic]

import os
import cv2 as cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg',force=True)

from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from datetime import datetime
from datetime import timedelta

# Directory structure
#
# object-detection 
#
#   + model   ssd_mobilenet_v1_coco_11_06_2017
#   + data    labels
#   + images  images for testing
#   + videos  videos for testing
#   + output  output from program
#

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_GRAPH = 'frozen_inference_graph.pb'
MODEL_LABEL = 'mscoco_label_map.pbtxt'

PATH_TO_CKPT = os.path.join(CWD_PATH, 'model', MODEL_NAME, MODEL_GRAPH)
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', MODEL_LABEL)

NUM_CLASSES = 90

# Load the label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Define all functions here

def duration_seconds():
    '''
       returns the elapsed duration in seconds since the start_time was last set
    '''
    global start_time

    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    ms = ms / 1000.0
    return ms

def duration(t,s):
    '''
      assume that start_time is set up for 1st call, uses duration(t,s) to calculate seconds
      t start - for starting the counter else end
      s string to display
    '''
    global start_time

    if (t=="start"):
        start_time = datetime.now()
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        s = "start "+s
        print("{0} >> {1}".format(s.rjust(30,' '),t))
    else:
        taken_time = duration_seconds()
        s = "end "+s
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{0} >> {1} taking {2:.2f} sec".format(s.rjust(30,' '), end_time, taken_time))
    return

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np

# IMAGE_SIZE = (12, 8)

# Load a frozen TF model 
# duration("start","loading detection graph")

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# duration("end","loading detection graph")

# detection starts now

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        
        plt.switch_backend('tkagg')
        # plt.ion()

        cam = cv2.VideoCapture(1) # modify here for camera number
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        while True:
            # Get camera frame
            ret, img = cam.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # duration("start","detect image")
            image_process = detect_objects(img, sess, detection_graph)
            # duration("end","detect image")
            image_process = cv2.cvtColor(image_process, cv2.COLOR_BGR2RGB)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(image_process, cv2.COLOR_BGR2RGB))
            plt.pause(0.001)
            # plt.waitforbuttonpress(0)

cam.release()
sess.close()
cv2.destroyAllWindows()        
        