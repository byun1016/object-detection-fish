import numpy as np
import time,datetime
import cv2
import argparse
import multiprocessing
import sys
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

from watchdog.events import *
import threading

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 90  #XX.pbtxt中类别的数目

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


input_q = Queue()
output_q = Queue()
logQueue = Queue()

class OwnVideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam网络摄像机, comment the line below out and use a video file
        # instead.
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('rtmp://live.hkstv.hk.lxdns.com/live/hks')
        #self.output_queue = output_q;
        print("init invoked!")


    def get_frame(self):
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        output_rgb = output_q.get()
        #print("get frame invoked")
        ret, jpeg = cv2.imencode('.jpg', output_rgb)
        return jpeg.tobytes()

count=0
boxes=[]
scores=[]
classes=[]
num_detections=[]
#图像检测，每帧中检测数目，框，类别，百分比。
def detect_objects(image_np, sess, detection_graph, logQueue):
    global boxes
    global scores
    global classes
    global num_detections
    global count
    count= count+1
    if (count % 10 != 0 and len(boxes) > 0):
        #print("do not return detect, just return original image")
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            #np.squeeze(boxes),#不标记框，防止每帧都输出坐标
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np

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


    print("start to detect object")
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    img, object_list = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    time_str = datetime.datetime.now()
    #str_total = "  ".join(object_list)
    str_entityList = []
    for objectDetail in object_list:
        entityList = objectDetail.split(":")
        if len(entityList) == 3:
            str_entityList.append(entityList[1])
            str_entityList.append(":")
            str_entityList.append(entityList[2])
            str_entityList.append(",")
    str_total = "".join(str_entityList)
    #print(time_str)     #输出当前时间
    print(str_total)     #输出类别以及百分比
    if len(str_total) > 0:
        logQueue.put(str_total)
    while (logQueue.qsize() > 200):
        logQueue.get()
    return image_np


def worker(input_q, output_q, logQueue):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)

    fps = FPS().start()
    while True:
        try:
            fps.update()
            frame = input_q.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#转换输出每帧的色彩
            output_q.put(detect_objects(frame_rgb, sess, detection_graph, logQueue))#将检测的每帧赋给输出
            pic = cv2.resize(output_q.get(), (1920, 1080), interpolation=cv2.INTER_CUBIC)  # 显示屏幕大小
            cv2.imshow('frame',pic)#展示检测后的输出
            if cv2.waitKey(1) == ord('q'):
                break
            while (output_q.qsize() > 200):
                output_q.get()
        except :
            #print("检测目标部分代码出现异常！")
            continue

    fps.stop()
    sess.close()


video_source=cv2.VideoCapture(0)   # 0 本地摄像头
#'rtmp://live.hkstv.hk.lxdns.com/live/hks'#网络直播     #'1.jpg'#图片    #'d.mp4'#本地视频
#rtsp://admin:PUHOU123@169.254.85.174:554/h264/ch1/main/av_stream
video_source_changed=False


def GetVideo(source,w, h):
    while True:
        global video_source
        global video_source_changed
        
        video_capture = video_source
        video_source_changed = False

        while True:  # fps._numFrames < 120
            if video_source_changed == True:
                print ('detect video_source is changed to %s, rebuild video source' %video_source)
                video_capture.release()
                break
            success, frame = video_capture.read()
            #print("get a frame")
            if success == True:
                input_q.put(frame)
            else:
                break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        default="", help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=100, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    get_video_thread = threading.Thread(target=GetVideo, args=(args.video_source, args.width, args.height))
    get_video_thread.start()

    pool = Pool(args.num_workers, worker, (input_q, output_q, logQueue))