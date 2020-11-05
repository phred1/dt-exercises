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
        self.distance = self.parameters['~look_ahead']
        self.follow_point = FollowPoint(1,0)
        self.do_pure_pursuit = True

    # Computes middle point of a segment
    def from_segment(self, segment):
        return self.middle(segment.points[0], segment.points[1])

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters
        self.distances = []

        self.previous_distances = []
        self.previous_follow_points = []
        self.left_turn = False
        self.set_follow_point_lists(self.parameters["~yellow_lines"], self.parameters["~white_lines"])
        self.set_distances()
        self.set_follow_point()

    # Computes middle point between two points
    def middle(self, point1, point2):
        y_weigth = 0.7
        w_weigth = 0.3

        x = (y_weigth * point1.x + w_weigth * point2.x) 
        y = (y_weigth * point1.y + w_weigth * point2.y)
        middle = FollowPoint(x, y)
        return middle

    # Sets the white and yellow list of possible follow points
    def set_follow_point_lists(self, segments_yellow, segments_white):
        # white
        possible_follow_points =  []
        for segment in segments_white:
            if segment.points[0].y <= 0:
                wp = self.from_segment(segment)
                if self.right_white_line(segments_yellow, wp):
                    possible_follow_points.append(wp)
        self.follow_points_white = possible_follow_points

        # yellow
        possible_follow_points =  []
        num_yellow_on_right = 0
        for segment in segments_yellow:
            if segment.points[0].y >= 0:
                yp = self.from_segment(segment)
                if self.not_grass(segments_white, yp):
                    possible_follow_points.append(yp)
            else:
                num_yellow_on_right += 1
        self.follow_points_yellow = possible_follow_points
        if num_yellow_on_right >= len(possible_follow_points) :
            self.left_turn = True
        else:
            self.left_turn = False

    # Check if the segments detected as yellow is not grass, by checking if the segment
    # is behind a white line
    def right_white_line(self, yellow_lines, white_point):
        if len(yellow_lines) == 0:
            return True
        d_total = 0
        for yl in yellow_lines:
            d = (white_point.x - yl.points[0].x)*(yl.points[1].y - yl.points[0].y) - (white_point.y - yl.points[0].y)*(yl.points[1].x - yl.points[0].x)
            if d < 0:
                return False
            d_total += d 
        mean_d = d_total/len(yellow_lines)
        return mean_d > 0 

    # Check if the segments detected as yellow is not grass, by checking if the segment
    # is behind a white line
    def not_grass(self, white_lines, yellow_point):
        if len(white_lines) == 0:
            return True
        d_total = 0
        for wl in white_lines:
            d = (yellow_point.x - wl.points[0].x)*(wl.points[1].y - wl.points[0].y) - (yellow_point.y - wl.points[0].y)*(wl.points[1].x - wl.points[0].x)
            d_total += d 
        mean_d = d_total/len(white_lines)
        return mean_d > 0 

    # Computes the distances between each segments' middle point, relative to the robot
    def set_distances(self):
        distances =  []
        wps = self.follow_points_white
        for point in wps:
            d = np.sqrt(point.x** 2 + point.y ** 2)
            distances.append(d)
        self.distances_white = np.asarray(distances)

        distances =  []
        yps = self.follow_points_yellow
        for point in yps:
            d = np.sqrt(point.x** 2 + point.y ** 2)
            distances.append(d)

        self.distances_yellow = np.asarray(distances)

    # Gets follow point which is the closest to the look_ahead distance 
    # from the list of possible follow_points
    def get_nearest_to_ref(self, distances, reference_dist, is_current):

        idx = (np.abs(distances - reference_dist)).argmin()
        return idx

    # Sets the follow point from computing the middle point between the white follow point and
    # the yellow follow point
    def set_follow_point(self):
        """
            Input:
                - follow_points: numpy array of possible follow points [x,y] in robot frame
            Return:
                - follow_point: numpty array of follow point closest to the ~look_ahead parameter
        """

        if self.distances_yellow.size > 0 and self.distances_white.size > 0:
            idx_yellow = self.get_nearest_to_ref(self.distances_yellow, self.parameters['~look_ahead'], True)
            follow_point_yellow = self.follow_points_yellow[idx_yellow]
            if self.distances_white.size > 0 :
                idx_white = self.get_nearest_to_ref(self.distances_white, self.parameters['~look_ahead'], True) 
                follow_point_white = self.follow_points_white[idx_white]

                self.follow_point = self.middle(follow_point_yellow, follow_point_white)
                self.distance = np.sqrt(self.follow_point.x** 2 + self.follow_point.y ** 2)

        else:
            self.do_pure_pursuit = False


    # Executes the pure_pursuit algorithm
    def pure_pursuit(self):
        """
        Input:
            - follow_points: numpy array of follow points [x,y] in robot frame
            - K: controller gain
        Return:
            - v: linear velocity in m/s (float)
            - w: angular velocity in rad/s (float)
        """

        sin_alpha =  self.follow_point.y / self.distance
        
        v = 0.1
        
        w = None
        if self.do_pure_pursuit:
            
            w = (sin_alpha / self.parameters["~K"])

        self.do_pure_pursuit = True

        return v, w