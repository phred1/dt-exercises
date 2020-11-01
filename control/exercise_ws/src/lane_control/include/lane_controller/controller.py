import numpy as np

class FollowPoint():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.distances_yellow = np.zeros(5)
        self.distances_white = np.zeros(5)
        self.distance = self.parameters['~follow_dist']
        self.follow_point = FollowPoint(1,0)
        self.is_pure = True
    def from_segment(self, segment):
        x = (segment.points[0].x + segment.points[1].x) / 2
        y = (segment.points[0].y + segment.points[1].y) / 2
        # print("x: " + str(x))
        # print("y: " + str(y))
        return  FollowPoint(x,y)

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters
        self.distances = []

        self.previous_distances = []
        self.previous_follow_points = []

        self.set_follow_point_list(self.parameters["~yellow_lines"], self.parameters["~white_lines"])

        self.set_distances()
        self.set_follow_point()
        
    def middle(self, point1, point2):
        x = (point1.x + point2.x) / 2
        y = (point1.y + point2.y) / 2
        middle = FollowPoint(x, y)
        return middle


    def set_follow_point_list(self, segments_yellow, segments_white):
        possible_follow_points =  []
        for segment in segments_yellow:
            if segment.points[0].y >= 0:
                possible_follow_points.append(self.from_segment(segment))
        self.follow_points_yellow = possible_follow_points
        # print("LEN_YELLOW")
        # print(len(self.follow_points_yellow))
        possible_follow_points =  []
        for segment in segments_white:
            if segment.points[0].y <= 0:
                possible_follow_points.append(self.from_segment(segment))
        self.follow_points_white= possible_follow_points

        # print("LEN_WHITE")
        # print(len(self.follow_points_white))


    def set_distances(self):

        distances =  []
        for point in self.follow_points_yellow:
            d = np.sqrt(point.x** 2 + point.y ** 2)
            distances.append(d)
        # print("D_YELLOW")
        self.distances_yellow = np.asarray(distances)
        # print(len(self.distances_yellow))
        distances =  []
        for point in self.follow_points_white:
            d = np.sqrt(point.x** 2 + point.y ** 2)
            distances.append(d)
        # print("D_WHITE")
        self.distances_white = np.asarray(distances)
        # print(len(self.distances_white))
    def get_nearest_to_ref(self, distances, reference_dist, is_current):
        if not is_current:
            reference_dist *= 1.5

        idx = (np.abs(distances - reference_dist)).argmin()
        return idx

    def set_follow_point(self):
        """
            Input:
                - follow_points: numpy array of possible follow points [x,y] in robot frame
            Return:
                - follow_point: numpty array of follow point closest to the ~follow_dist parameter
        """
        # self.previous_distance = self.distance
        # self.previous_follow_point = self.follow_point

        if self.distances_yellow.size > 0 and self.distances_white.size > 0 :
            # print("CURRENT")
            idx_yellow = self.get_nearest_to_ref(self.distances_yellow, self.parameters['~follow_dist'], True)
            follow_point_yellow = self.follow_points_yellow[idx_yellow]
            # print("FOLLOW_POINT_YELLOW")
            # print(follow_point_yellow.x)
            idx_white = self.get_nearest_to_ref(self.distances_white, self.parameters['~follow_dist'], True)
            follow_point_white = self.follow_points_white[idx_white]
            # print("FOLLOW_POINT_WHITE")
            # print(follow_point_white.x)
            self.follow_point = self.middle(follow_point_yellow, follow_point_white)
            self.distance = np.sqrt(self.follow_point.x** 2 + self.follow_point.y ** 2)
        else:
            self.is_pure = False
        #     print("PREVIOUS")
        #     self.distance = self.previous_distance
        #     self.follow_point = self.previous_follow_point


    def pure_pursuit(self):
        """
        Input:
            - follow_points: numpy array of follow points [x,y] in robot frame
            - K: controller gain
        Return:
            - v: linear velocity in m/s (float)
            - w: angular velocity in rad/s (float)
        """

        # compute distance between robot and follow point
        # print("DISTANCE")
        # print(self.distance)
        # TODO: compute sin(alpha)
        sin_alpha =  self.follow_point.y / self.distance
        
        v = 0.1 # we can make it constant or we can make it as a function of sin_alpha
        
        # TODO: compute angular velocity
        w = None
        if self.is_pure:
            w = sin_alpha / self.parameters["~K"]

        self.is_pure = True

        return v, w