# reading the individual frames of the video

import cv2

def readframes(video_path):
    frames=[]
    cap = cv2.VideoCapture(video_path)
    
    while True:
        
        ret , frame = cap.read()
        
        if not ret :
            break
        frames.append(frame)
        
    cap.release()
    
    return frames

def save_video( output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()