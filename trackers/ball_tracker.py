from ultralytics import YOLO

import cv2
import pickle 
import pandas as pd

class Ball_Tracker:
    def __init__(self, model_path):
        
        self.model=YOLO(model_path)
        
    
    # this will track only the person out of the frame and not any other class
    
    def readframes(self, frame):
        
        results = self.model.predict(frame,conf=0.2)[0]
        
        ball_track_dict = {}
        
        for box in results.boxes :
            
            result_box = box.xyxy.tolist()[0]
                
            ball_track_dict[1] = result_box
                
        return ball_track_dict
    
    # now to detect for multiple frames
    
    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        
        ball_detection=[]
        
        if read_from_stub==True  and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detection = pickle.load(f)
                
                return ball_detection
            
        for frame in frames:
            each_frame_detection = self.readframes(frame)
            
            ball_detection.append(each_frame_detection)
            
        if stub_path is not None :
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detection,f)
            
        return ball_detection
    
    def interpolate_frames(self, ball_positions):
        
        ball_positions = [x.get(1,[]) for x in ball_positions]
        
        #converting into pandas dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        #interpolating using builtin function
        ball_positions_df = ball_positions_df.interpolate()
        #now there  may be a case where the first element of the df is missing so to avoid that
        ball_positions_df = ball_positions_df.bfill()
        
        #now converting back to the original data
        
        ball_positions_interpolated = [{1 : x} for x in ball_positions_df.to_numpy().tolist()]
        
        return ball_positions_interpolated
    
    def get_hit_frame(self, ball_positions):
         #converting into pandas dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        ball_positions_df['mid_y'] = (ball_positions_df['y1'] + ball_positions_df['y2'])/2

        ball_positions_df['rolling_mid_y'] = ball_positions_df['mid_y'].rolling( window=5 ,min_periods=1, center=False).mean()

        min_change_in_frame = 20

        ball_positions_df['hit'] = 0

        for i in range(len(ball_positions_df) - 1) :
            positive_change = ball_positions_df['difference_y'].iloc[i] < 0  and ball_positions_df['difference_y'].iloc[i+1] > 0
            negative_change = ball_positions_df['difference_y'].iloc[i] > 0  and ball_positions_df['difference_y'].iloc[i+1] < 0
            
            hit_count = 0
            
            if positive_change or negative_change :
                
                
                for each_frame in range(i, i + int(min_change_in_frame*1.2)):
                    
                    pos_change = ball_positions_df['difference_y'].iloc[i] < 0  and ball_positions_df['difference_y'].iloc[each_frame+1] > 0
                    neg_change = ball_positions_df['difference_y'].iloc[i] > 0  and ball_positions_df['difference_y'].iloc[each_frame+1] < 0
                    
                    if(pos_change and positive_change) or (neg_change and negative_change): 
                        hit_count+=1 
                        #print('hey')
                
                if hit_count >= min_change_in_frame-1 :
                    ball_positions_df['hit'].iloc[i] = 1
                    
        hit_frame = ball_positions_df[ball_positions_df['hit']==1].index.tolist()
        
        return hit_frame
            
    def draw_bboxes(self ,video_frames, ball_detection):
        
        output_video_frames = []
        
        for frame , ball_track_dict in zip(video_frames, ball_detection):
            
            for track_id , bbox in ball_track_dict.items():
                
                x1, y1, x2, y2 = bbox
                
                frame = cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (10, 10, 0), 2)
                
            output_video_frames.append(frame)
        
        return output_video_frames