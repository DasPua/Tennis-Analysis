from ultralytics import YOLO

import cv2
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_center,measure_distance

class Player_Tracker:
    def __init__(self, model_path):
        
        self.model=YOLO(model_path)
        
    # this will track only the person out of the frame and not any other class
    
    def readframes(self, frame):
        
        results = self.model.track(frame, persist= True)[0]
        
        id_dict = results.names
        
        player_track_dict = {}
        
        for box in results.boxes :
            
            box_id = int(box.id.tolist()[0])
            
            result_box = box.xyxy.tolist()[0]
            
            class_id = box.cls.tolist()[0]
            
            class_name = id_dict[class_id]
            
            if class_name == "person":
                
                player_track_dict[box_id] = result_box
                
        return player_track_dict
    
    # now to detect for multiple frames
    
    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        
        player_detection=[]
        
        if read_from_stub==True  and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detection = pickle.load(f)
                
                return player_detection
            
        for frame in frames:
            each_frame_detection = self.readframes(frame)
            
            player_detection.append(each_frame_detection)
            
        if stub_path is not None :
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detection,f)
            
        return player_detection
            
    def draw_bboxes(self ,video_frames, player_detection):
        
        output_video_frames = []
        
        for frame , player_track_dict in zip(video_frames, player_detection):
            
            for track_id , bbox in player_track_dict.items():
                
                x1, y1, x2, y2 = bbox
                
                frame = cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
            output_video_frames.append(frame)
        
        return output_video_frames
    
    def choose_player(self, player_detection_dict, court_keypoints):
        
        distances = []
        for track_id , bbox in player_detection_dict.items():
            curr_distance = float('inf')
            center =get_center(bbox)
        
            for i in range(0 , len(court_keypoints), 2):
                coordinates = (court_keypoints[i] , court_keypoints[i+1])
                
                distance = measure_distance(center, coordinates)
                if distance < curr_distance:
                    curr_distance = distance
            
            distances.append((curr_distance, track_id))
        
        distances.sort(key = lambda x : x[0])
        
        # print(distances)
        
        return [distances[0][1] , distances[1][1]]
    
    def choose_and_filter_players(self, court_keypoints, player_detection):
        first_frame = player_detection[0]
        
        chosen_player = self.choose_player(first_frame , court_keypoints)## this will only give the player id
        
        player_id_map = {chosen_player[0]: 1, chosen_player[1]: 2}
        
        filtered_players=[]
        
        # print(chosen_player)
        
        for player_dict in player_detection: ## loop through all the player-bbox dictionaries
            
            chosen_player_dict  = { player_id_map[track_id] : bbox for track_id ,bbox in player_dict.items() if track_id in chosen_player }

            #upar wala will give the bounding boxex and player id of chosen player
            
            filtered_players.append(chosen_player_dict)
        
        # for frame_num, players in enumerate(filtered_players):
        #     print(f"Frame {frame_num}: {players}")
        
        return filtered_players