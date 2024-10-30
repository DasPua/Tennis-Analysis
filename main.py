from utils import (readframes,
                   save_video,
                   )

from trackers import (Player_Tracker,
                      Ball_Tracker)

from court_line_detectors import (CourtLineDetector)

import cv2

from mini_court import mini_court_display

def main():
    
    #reading the video
    video_frames = readframes('input_files//input_video.mp4')
    
    #detecting the players and the ball
    player_tracking = Player_Tracker(model_path = 'yolov5xu.pt')
    ball_tracking = Ball_Tracker(model_path = 'models\\yolov5xu_last.pt')
    
    player_detection = player_tracking.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path = 'trackers_stubs//player_detections.pkl')  #this wont draw the boxes on the video
    
    ball_detection = ball_tracking.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path = 'trackers_stubs//ball_detections.pkl')  #this wont draw the boxes on the video
    
    #doing the interpolation
    ball_detection = ball_tracking.interpolate_frames(ball_detection)
    
    #court line prediction
    
    Courtlinedetector = CourtLineDetector('models\\keypoints_model.pth')
    
    courtkeypoints  = Courtlinedetector.predict(video_frames[0])
    
    #filtering the players i.e. only the ones closest to the court 
    
    filtered_player_detection = player_tracking.choose_and_filter_players(courtkeypoints, player_detection)
    
    # drawing the bboxes
    
    output = player_tracking.draw_bboxes(video_frames, filtered_player_detection)
    output = ball_tracking.draw_bboxes(output, ball_detection)
    
    #drawing the court lines
    
    output = Courtlinedetector.draw_keypoints_on_video(output, courtkeypoints)
    
    #drawing the mini court
    mini_court = mini_court_display(video_frames[0])
    output = mini_court.draw_mini_court(output)
    
    #converting the bbox positions to mini court positions
    
    player_mini_court_positions, ball_mini_court_positions = mini_court.convert_bounding_boxes_to_mini_court_coordinates(filtered_player_detection,
                                                                                                                         ball_detection,
                                                                                                                         courtkeypoints)
    #drawing the ball and the players
    output = mini_court.draw_positions_on_mini_court(output, ball_mini_court_positions, color=(255,0,255))
    output = mini_court.draw_positions_on_mini_court(output, player_mini_court_positions)
     ## Draw frame number on top left corner
    for i, frame in enumerate(output):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    save_video(output, 'output_files//test_6.avi')
    
if __name__ == "__main__":
    main()