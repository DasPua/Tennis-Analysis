import cv2
import sys
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants
from utils import (convert_pixel_distance_to_meters,
                   convert_meters_to_pixel_distance,
                   get_foot_positions,
                   get_closest_keypoint_index,
                   get_height_of_bbox,
                   measure_xy_distance,
                   measure_distance) 

class mini_court_display:
    def __init__(self, frame):
        self.display_height = 600
        self.display_width = 300
        self.buffer = 30
        self.padding = 20
        
        self.display_background_box_position(frame)
        self.mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        
    def display_background_box_position(self, frame):
        frame = frame.copy()
        
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.display_height + self.buffer
        self.start_x = self.end_x - self.display_width
        self.start_y = self.end_y - self.display_height
        
    def mini_court_position(self):
        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y + self.padding
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y - self.padding
        self.court_width = self.end_x - self.start_x
        
        # Set court drawing width for pixel conversion
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                                )
        
    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28

        # point 0 
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4], drawing_key_points[5] = int(self.court_start_x), int(self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2))
        # point 3
        drawing_key_points[6], drawing_key_points[7] = int(self.court_end_x), drawing_key_points[5]
        # point 4
        drawing_key_points[8], drawing_key_points[9] = int(self.court_start_x + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)), drawing_key_points[1]
        # point 5
        drawing_key_points[10], drawing_key_points[11] = int(drawing_key_points[8]), drawing_key_points[5]
        # point 6
        drawing_key_points[12], drawing_key_points[13] = int(self.court_end_x - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)), drawing_key_points[3]
        # point 7
        drawing_key_points[14], drawing_key_points[15] = int(drawing_key_points[12]), drawing_key_points[7]
        # point 8
        drawing_key_points[16], drawing_key_points[17] = drawing_key_points[8], int(drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT))
        # point 9
        drawing_key_points[18], drawing_key_points[19] = int(drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)), drawing_key_points[17]
        # point 10
        drawing_key_points[20], drawing_key_points[21] = drawing_key_points[10], int(drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT))
        # point 11
        drawing_key_points[22], drawing_key_points[23] = int(drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)), drawing_key_points[21]
        # point 12
        drawing_key_points[24], drawing_key_points[25] = int((drawing_key_points[16] + drawing_key_points[18]) / 2), drawing_key_points[17]
        # point 13
        drawing_key_points[26], drawing_key_points[27] = int((drawing_key_points[20] + drawing_key_points[22]) / 2), drawing_key_points[21]

        self.drawing_key_points = drawing_key_points
    
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (12, 13),  # Changed to (12, 13) to avoid duplicate
            (2, 3)
        ]

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_court(self, frame):
        
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)
            
        # Draw the lines 
        for line in self.lines:
            pt1 = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            pt2 = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, pt1, pt2, (0, 0, 0), 2)  
        #net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)
        
        return frame
      
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_starting_points(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_court_width(self):
        return self.court_width
    
    def get_drawing_keypoints(self):
        return self.drawing_key_points
    
    #scrapped part
    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_court_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_court_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_court_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_bounding_boxes, ball_bounding_box, court_key_points ):
        # Convert the bounding boxes to mini court coordinates by cross multiplying the pixel height of the player with the pixel distance from the court keypoints
        
        output_player_bbox = []
        output_ball_bbox = []
        
        player_heights ={
            1 : constants.PLAYER_1_HEIGHT_METERS,
            2 : constants.PLAYER_2_HEIGHT_METERS
        }
        
        for frame_no , player_bbox in enumerate(player_bounding_boxes):
            output_player_bbox_dict = {}
            
            ball_bbox = ball_bounding_box[frame_no][1]
            ball_bbox_center = (int((ball_bbox[0]+ball_bbox[2])/2),int((ball_bbox[1]+ball_bbox[3])/2))
            
            closest_player_bbox_id = min(player_bbox.keys() , key = lambda x : measure_distance(ball_bbox_center,(int((player_bbox[x][0]+player_bbox[x][2])/2),int((player_bbox[x][1]+player_bbox[x][3])/2)) ))
            for player_id, bbox in player_bbox.items():
                 ## we need the distance of the bottom of the box of the player
                foot_positions = get_foot_positions(bbox)
                 
                 #now getting the closest key point
                 
                closest_key_index = get_closest_keypoint_index(foot_positions, court_key_points, [0,2,12,13])
                 
                closest_key_points = (court_key_points[closest_key_index*2], court_key_points[closest_key_index*2+1])
                 
                 #now to get the height of the bbox in pixels :
                 
                min_frame = max(0, frame_no-20)
                max_frame = min(frame_no+50, len(player_bounding_boxes))
                 
                max_height_in_pixels = max([get_height_of_bbox(player_bounding_boxes[i][player_id]) for i in range(min_frame, max_frame)])
                 
                mini_court_player_position = self.get_mini_court_coordinates(foot_positions,
                                                                            closest_key_points, 
                                                                            closest_key_index, 
                                                                            max_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                output_player_bbox_dict[player_id] = mini_court_player_position
                
                if closest_player_bbox_id == player_id:
                    closest_key_index = get_closest_keypoint_index(ball_bbox_center, court_key_points, [0,2,12,13])
                 
                    closest_key_points = (court_key_points[closest_key_index*2], court_key_points[closest_key_index*2+1])
                    
                    mini_court_ball_position = self.get_mini_court_coordinates(ball_bbox_center,
                                                                            closest_key_points, 
                                                                            closest_key_index, 
                                                                            max_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_bbox.append({1: mini_court_ball_position})
            output_player_bbox.append(output_player_bbox_dict)
            
        return output_player_bbox,output_ball_bbox
    
    def draw_positions_on_mini_court(self, frames,positions, color = (0,255,100)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames