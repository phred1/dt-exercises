#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, AntiInstagramThresholds
from image_processing.anti_instagram import AntiInstagram
import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge

class Point:
    """
    Point class. Convenience class for storing ROS-independent 3D points.

    """
    def __init__(self, x=None, y=None):
        self.x = x  #: x-coordinate
        self.y = y  #: y-coordinate

class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )



        # Construct publishers
        self.pub_obj_dets = rospy.Publisher(
            "~duckie_detected",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )
        
        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )
        
        self.ai_thresholds_received = False
        self.anti_instagram_thresholds=dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()

        model_file = rospy.get_param('~model_file','.')
        rospack = rospkg.RosPack()
        model_file_absolute = rospack.get_path('object_detection') + model_file
        self.model_wrapper = Wrapper(model_file_absolute)
        self.initialized = True
        self.log("Initialized!")
    
    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg):
        if not self.initialized:
            return

        # TODO to get better hz, you might want to only call your wrapper's predict function only once ever 4-5 images?
        # This way, you're not calling the model again for two practically identical images. Experiment to find a good number of skipped
        # images.

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return
        
        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )
        
        image = cv2.resize(image, (224,224))
        bboxes, classes, scores = self.model_wrapper.predict(image)
        
        msg = BoolStamped()
        msg.header = image_msg.header
        msg.data = self.det2bool(bboxes[0], classes[0]) # [0] because our batch size given to the wrapper is 1
        
        self.pub_obj_dets.publish(msg)

    def midpoint(self, p1, p2):
        return Point((p1.x+p2.x)/2, (p1.y+p2.y)/2)
    
    def det2bool(self, bboxes, classes):
    
        middle_bounds_x = (112-20, 112 + 20)
        middle_bounds_y = (112-20, 112 + 20)
        for i in range(len(bboxes)):
            if abs(bboxes[i][0] - bboxes[i][2]) < 5 or abs(bboxes[i][1] - bboxes[i][3]) < 5:
               print("SKIP")
               continue

            if classes[i] == 1:
                print("DUCKIE")
                lower = Point(bboxes[i][0], bboxes[i][1]) 
                upper = Point(bboxes[i][2], bboxes[i][3]) 
                middle = self.midpoint(lower, upper)
                if middle.x > middle_bounds_x[0] and middle.x < middle_bounds_x[1]:
                     if middle.y > middle_bounds_y[0] and middle.y < middle_bounds_y[1]: 
                        print("IN-FRONT")
                        return True
        return False




if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
