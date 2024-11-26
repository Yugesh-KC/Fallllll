import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import cv2
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the pre-trained model
model = models.video.r3d_18(pretrained=True)  # R3D model pretrained on Kinetics-400
model.fc = nn.Linear(model.fc.in_features, 2)  # Replace the final layer with another classification layer

# Freeze the pretrained weights
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():  # We need to train the new weights
    param.requires_grad = True

# Load the state_dict for the model
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((112, 112)),  # Resize frame to 112x112
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to prepare video from frames
def prepare_video(video_path, num_frames=32):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(240)  #8 sec * 30 fps = 240 frames 
    
    

    
    step=int(240/(num_frames-1))     # to calculate which frames to include (we only retain 32 frames from a 10 sec)

    for idx in range(total_frames):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        
        if idx%step==0:   #we need only 32 frames
        

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB cuz open cv for some reason loads in bgr
            frames.append(transform(frame))        
        
        

    
    cap.release()

    # Stack frames and format for the model
    video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
    return video_tensor.unsqueeze(0)  # Add batch dimension: [C, T, H, W] -> [B, C, T, H, W]

# Function to classify video using the model
def classify_video_from_stream(stream_url, model):
      
    input_tensor = prepare_video(stream_url)
    input_tensor = input_tensor.to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # Output shape: [B, num_classes]
        probabilities = torch.softmax(output, dim=1)
        probability, predicted_class_idx = torch.max(probabilities, dim=1)

    return predicted_class_idx.item(),probability.item()

if __name__ == "__main__":
    # Replace with your stream URL
    stream_url = "http://192.168.18.84:8080/video"
    
    while True:
        print(classify_video_from_stream(stream_url, model))
        
