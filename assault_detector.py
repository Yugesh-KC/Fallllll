import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.video.r3d_18(pretrained=True)  # R3D model pretrained on Kinetics-400
model.fc = nn.Linear(model.fc.in_features, 3) #Replace the final layer with another classification layer

for param in model.parameters():
    param.requires_grad=False        #no need to train the pretrained weights

for param in model.fc.parameters():   #we need to train the new weights 
    param.requires_grad = True
    
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((112, 112)),  # Resize frame to 112x112
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to extract and preprocess frames
def prepare_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(transform(frame))
    
    cap.release()

    # Stack frames and format for the model
    video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
    return video_tensor.unsqueeze(0)  # Add batch dimension: [C, T, H, W] -> [B, C, T, H, W]

def classify_video(video_path, model):
    # Prepare the video as input
    model.eval()
    input_tensor = prepare_video(video_path)
    
    input_tensor=input_tensor.to(device)
    print(input_tensor.type())

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)  # Output shape: [B, num_classes]
        
        print(output)

    
    # Get predicted class index
    predicted_class_idx = torch.argmax(output,dim=1).item()
    # predicted_label = class_labels[predicted_class_idx]
    return predicted_class_idx



# Classify the video
video_path = "beating.mp4"   #near 10 sec video clip
label = classify_video(video_path, model)

labels={0:"non assault", 1:"alert",2:"assault"}
print(labels[label])


