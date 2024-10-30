import torch
import torchvision.models as models
import torchvision.transforms as transforms

import cv2

class CourtLineDetector:
    def __init__(self,model_path):
        
        self.model= models.resnet50(pretrained=True)
        
        self.model.fc=torch.nn.Linear(self.model.fc.in_features, 14*2)
        
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.transforms= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize( mean=[.4, .4, .4] , std=[.2, .2, .2])
        ])
        
    def predict(self, image):
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_tensor = self.transforms(image).unsqueeze(0) #makes the image into another list i.e. [image] because the models need in that format
        
        with torch.no_grad():
            output=self.model(image_tensor)
            
            keypoints= output.squeeze().cpu().numpy()
            
            actual_h, actual_w = image.shape[:2]
            
            keypoints[::2] *= actual_w / 224.0
            
            keypoints[1::2] *= actual_h / 224.0
            
        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
    
        