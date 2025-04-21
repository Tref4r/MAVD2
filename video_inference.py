import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dataset.dataset_loader import Dataset
from model.unimodal import Unimodal
from model.multimodal import Multimodal
from model.projection import Projection
from utils.utils import process_feat
import matplotlib.pyplot as plt

def extract_frames(video_path, frame_skip=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (224, 224))  # Resize to model input size
            frames.append(frame)
        frame_count += 1

    cap.release()
    return np.array(frames)

def preprocess_frames(frames):
    # Normalize and reshape frames for the model
    frames = frames / 255.0  # Normalize to [0, 1]
    frames = np.transpose(frames, (0, 3, 1, 2))  # Convert to (N, C, H, W)
    return frames.astype(np.float32)

def extract_features(frames):
    # Use a pre-trained ResNet model to extract features
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()  # Remove the classification layer
    resnet = resnet.cuda().eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((224, 224)),  # Ensure the input size is 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    features = []
    with torch.no_grad():
        for frame in frames:
            frame_tensor = transform(frame).unsqueeze(0).cuda()  # Add batch dimension
            feature = resnet(frame_tensor)
            features.append(feature.squeeze(0).cpu().numpy())  # Remove the extra batch dimension

    return np.array(features)

def load_models(model_path):
    v_net = Unimodal(input_size=1024, h_dim=128, feature_dim=128).cuda()
    a_net = Unimodal(input_size=128, h_dim=64, feature_dim=32).cuda()
    f_net = Unimodal(input_size=1024, h_dim=128, feature_dim=64).cuda()
    p_net = Unimodal(input_size=512, h_dim=128, feature_dim=64).cuda()
    va_net = Projection(32, 32, 32).cuda()
    vf_net = Projection(64, 64, 64).cuda()
    vp_net = Projection(64, 64, 64).cuda()
    vafp_net = Multimodal(input_size=128 + 32 + 64 + 64, h_dim=128, feature_dim=64).cuda()

    v_net.load_state_dict(torch.load(f"{model_path}/v_model.pth"))
    a_net.load_state_dict(torch.load(f"{model_path}/a_model.pth"))
    f_net.load_state_dict(torch.load(f"{model_path}/f_model.pth"))
    p_net.load_state_dict(torch.load(f"{model_path}/p_model.pth"))
    va_net.load_state_dict(torch.load(f"{model_path}/va_model.pth"))
    vf_net.load_state_dict(torch.load(f"{model_path}/vf_model.pth"))
    vp_net.load_state_dict(torch.load(f"{model_path}/vp_model.pth"))
    vafp_net.load_state_dict(torch.load(f"{model_path}/vafp_model.pth"))

    return v_net, a_net, f_net, p_net, va_net, vf_net, vp_net, vafp_net

def infer_on_video(video_path, model_path, frame_skip=16):
    frames = extract_frames(video_path, frame_skip)
    features = extract_features(frames)  # Extract features from frames

    # Add a linear projection layer to reduce feature dimensionality
    projection_layer = torch.nn.Linear(2048, 1024).cuda()
    features = torch.tensor(features).cuda()
    features = projection_layer(features)

    v_net, a_net, f_net, p_net, va_net, vf_net, vp_net, vafp_net = load_models(model_path)

    with torch.no_grad():
        v_net.eval()
        a_net.eval()
        f_net.eval()
        p_net.eval()
        va_net.eval()
        vf_net.eval()
        vp_net.eval()
        vafp_net.eval()

        # Print shape for debugging
        print("Shape of projected features:", features.shape)

        # Reshape to (batch_size, sequence_length, feature_dim)
        batch_size = 1  # Single video inference
        sequence_length = features.shape[0]
        feature_dim = features.shape[1]
        rgb_features = features.view(batch_size, sequence_length, feature_dim)

        # Create placeholder tensors with correct dimensions
        audio_features = torch.zeros((batch_size, sequence_length, 128)).cuda()  # For a_net
        flow_features = torch.zeros((batch_size, sequence_length, 1024)).cuda()  # For f_net
        pose_features = torch.zeros((batch_size, sequence_length, 512)).cuda()  # For p_net

        # Perform inference
        v_res = v_net(rgb_features)
        a_res = a_net(audio_features)
        f_res = f_net(flow_features)
        p_res = p_net(pose_features)

        mix_f = torch.cat([v_res["satt_f"], va_net(a_res["satt_f"]), vf_net(f_res["satt_f"]), vp_net(p_res["satt_f"])], dim=-1)
        m_out = vafp_net(mix_f)

        predictions = m_out["output"].cpu().numpy()
        return predictions

def visualize_predictions(frames, predictions):
    """
    Visualize video frames with predictions overlaid.
    """
    for i, frame in enumerate(frames):
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
        plt.title(f"Frame {i + 1} - Prediction: {predictions[i]:.2f}", fontsize=16)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    video_path = r"c:\Users\tatra\Downloads\Man gets blown away by gasoline fire.mp4"
    model_path = "saved_models/84.98/"
    predictions = infer_on_video(video_path, model_path)

    # Extract frames for visualization
    frames = extract_frames(video_path, frame_skip=16)

    # Visualize predictions on frames
    visualize_predictions(frames, predictions)
