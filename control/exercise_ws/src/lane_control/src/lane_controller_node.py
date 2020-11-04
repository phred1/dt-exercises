#!/usr/bin/env python3
import numpy as np
import rospy
import inspect
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import SegmentList, Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~seglist_filtered (:obj:``SegmentList): Filtered list of segments that are considered as valid
        ~lineseglist_out (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in the ground plane relative to the robot origin
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.params['~look_ahead'] = rospy.get_param('~look_ahead', None)
        self.params['~K'] = rospy.get_param('~K', None)

        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        # self.log("INIT SEGLIST FILTERED")
        # self.sub_seglist_filtered = rospy.Subscriber("/agent/lane_filter_node/seglist_filtered",
        #                                          SegmentList,
        #                                          self.cbSeglistFiltered,
        #                                          queue_size=1)
        self.log("INIT SEGLIST OUT")                                     
        self.sub_seglist_filtered = rospy.Subscriber("/agent/ground_projection_node/lineseglist_out",
                                                 SegmentList,
                                                 self.cbSeglistOut,
                                                 queue_size=1)
        self.log("Initialized!")


    # def cbSeglistFiltered(self, input_seglist_filtered):
    #     """Callback receiving pose messages

    #     Args:
    #         input_seglist_filtered (:obj:`SegmentList`): Message containing information about filtered list of segments that are considered as valid.
    #     """
    #     yellow_lines = []
    #     white_lines = []
    #     for line in input_seglist_filtered.segments:
    #         if line.color == 1:
    #             yellow_lines.append(line)
    #         elif line.color == 0:
    #             white_lines.append(line) 
    #     self.params["~yellow_lines: "] = yellow_lines
    #     self.params["~white_lines"] = white_lines
    #     self.cbParametersChanged()
    

    def cbSeglistOut(self, input_seglist_out):
        """Callback receiving pose messages

        Args:
            input_seglist_out(:obj:`SegmentList`): Message containing information line segments in the ground plane relative to the robot origin.
        """

        yellow_lines = []
        white_lines = []
        for line in input_seglist_out.segments:
            if line.color == 1:
                yellow_lines.append(line)
            elif line.color == 0:
                white_lines.append(line) 

        self.params["~yellow_lines"] = yellow_lines
        self.params["~white_lines"] = white_lines
        self.cbParametersChanged()
    

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Line segments in the ground plane relative to the robot origin.
        """

        self.pose_msg = input_pose_msg
        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        v, omega = self.pp_controller.pure_pursuit()
        car_control_msg.v = v

        if omega :
            car_control_msg.omega = omega
        else:
            w = np.sin(self.pose_msg.phi + np.pi)
            w = 7 * w + np.sign(w) * 6 * self.pose_msg.d
            car_control_msg.omega = w
    
        self.publishCmd(car_control_msg)


    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.pp_controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
