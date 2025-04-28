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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import torch.backends.cudnn as cudnn

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

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
        for i, frame in enumerate(frames):
            try:
                frame_tensor = transform(frame).unsqueeze(0).cuda()  # Add batch dimension
                feature = resnet(frame_tensor)
                features.append(feature.squeeze(0).cpu().numpy())  # Remove the extra batch dimension
            except Exception as e:
                print(f"Error processing frame {i}: {e}")

    if len(features) == 0:
        raise RuntimeError("No features were extracted. Check the input frames or the ResNet model.")

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

def load_ground_truth(gt_path, frame_skip, num_predictions):
    """
    Load ground truth labels from a .npy file and subsample them to match the frame skip.
    """
    ground_truth = np.load(gt_path)
    subsampled_gt = ground_truth[::frame_skip]  # Subsample ground truth to match predictions

    # Ensure the subsampled ground truth matches the number of predictions
    if len(subsampled_gt) > num_predictions:
        subsampled_gt = subsampled_gt[:num_predictions]
    elif len(subsampled_gt) < num_predictions:
        raise ValueError(f"Mismatch between predictions ({num_predictions}) and subsampled ground truth ({len(subsampled_gt)}).")

    return subsampled_gt

def evaluate_predictions(predictions, ground_truth, threshold=0.5):
    """
    Evaluate predictions against ground truth using accuracy, precision, recall, and F1-score.
    """
    binary_predictions = (predictions > threshold).astype(int)
    accuracy = accuracy_score(ground_truth, binary_predictions)
    precision = precision_score(ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(ground_truth, binary_predictions, zero_division=0)
    return accuracy, precision, recall, f1

def normalize_tensor(tensor):
    """
    Normalize a tensor to have values between 0 and 1.
    """
    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return (tensor - min_val) / (max_val - min_val + 1e-8)

def infer_on_video(video_path, model_path, gt_path, frame_skip=16):
    """
    Perform inference on a video and evaluate predictions.
    """
    set_seed(42)  # Ensure deterministic behavior

    frames = extract_frames(video_path, frame_skip)
    if len(frames) == 0:
        raise RuntimeError("No frames were extracted from the video. Check the video path or frame extraction logic.")

    print(f"Number of frames extracted: {len(frames)}")

    features = extract_features(frames)  # Extract features from frames
    print(f"Shape of extracted features: {features.shape}")

    # Add a linear projection layer to reduce feature dimensionality
    projection_layer = torch.nn.Linear(2048, 1024).cuda()
    features = torch.tensor(features, dtype=torch.float32).cuda()
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

        print("Shape of projected features:", features.shape)

        # Reshape to (batch_size, sequence_length, feature_dim)
        batch_size = 1
        sequence_length = features.shape[0]
        feature_dim = features.shape[1]
        rgb_features = features.view(batch_size, sequence_length, feature_dim)

        # Create placeholder tensors with correct dimensions
        audio_features = torch.zeros((batch_size, sequence_length, 128), dtype=torch.float32).cuda()
        flow_features = torch.zeros((batch_size, sequence_length, 1024), dtype=torch.float32).cuda()
        pose_features = torch.zeros((batch_size, sequence_length, 512), dtype=torch.float32).cuda()

        # Perform inference
        v_res = normalize_tensor(v_net(rgb_features)["satt_f"])
        a_res = normalize_tensor(a_net(audio_features)["satt_f"])
        f_res = normalize_tensor(f_net(flow_features)["satt_f"])
        p_res = normalize_tensor(p_net(pose_features)["satt_f"])

        mix_f = torch.cat([v_res, va_net(a_res), vf_net(f_res), vp_net(p_res)], dim=-1)
        m_out = vafp_net(mix_f)

        predictions = m_out["output"].cpu().numpy()

        # Normalize predictions
        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        # Load and subsample ground truth
        ground_truth = load_ground_truth(gt_path, frame_skip, len(predictions))

        # Evaluate predictions
        accuracy, precision, recall, f1 = evaluate_predictions(predictions, ground_truth)
        print(f"Evaluation Metrics:\n"
              f"Accuracy: {accuracy:.4f}\n"
              f"Precision: {precision:.4f}\n"
              f"Recall: {recall:.4f}\n"
              f"F1-Score: {f1:.4f}")

        return predictions, frames

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

def plot_predictions_with_graph(frames, predictions, frame_skip, output_path):
    """
    Plot anomaly scores over time with 3 frames from the beginning, middle, and end of the video.
    Each frame in the beginning, middle, and end sections is spaced 5 frames apart.
    Frames are displayed in a single row, aligned in chronological order.
    """
    plt.figure(figsize=(18, 6))

    # Plot the anomaly scores
    time = np.arange(len(predictions)) * frame_skip
    plt.plot(time, predictions, label="Anomaly Score", color="blue", linewidth=2)

    # Highlight regions with high and low anomaly scores
    threshold = 0.5
    high_anomaly_indices = np.where(predictions > threshold)[0]
    low_anomaly_indices = np.where(predictions <= threshold)[0]

    # Highlight low anomaly regions
    for idx in low_anomaly_indices:
        plt.axvspan(time[idx] - frame_skip / 2, time[idx] + frame_skip / 2, color="green", alpha=0.1, label="Low Anomaly" if idx == low_anomaly_indices[0] else "")

    # Highlight high anomaly regions
    for idx in high_anomaly_indices:
        plt.axvspan(time[idx] - frame_skip / 2, time[idx] + frame_skip / 2, color="red", alpha=0.3, label="High Anomaly" if idx == high_anomaly_indices[0] else "")

    # Select 3 frames from the beginning, middle, and end, spaced 5 frames apart
    num_frames = len(frames)
    selected_indices = [
        0, min(5, num_frames - 1), min(10, num_frames - 1),  # First 3 frames spaced 5 apart
        max(num_frames // 2 - 5, 0), num_frames // 2, min(num_frames // 2 + 5, num_frames - 1),  # Middle 3 frames spaced 5 apart
        max(num_frames - 11, 0), max(num_frames - 6, 0), num_frames - 1  # Last 3 frames spaced 5 apart
    ]

    # Overlay frames at specific points
    for i, idx in enumerate(selected_indices):
        frame = frames[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Place frames in a single row at the center of the plot
        y_position = 0.5  # Center of the plot
        inset_ax = plt.axes([0.1 + (i / len(selected_indices)) * 0.8, y_position, 0.08, 0.15])  # Adjust width and height
        inset_ax.imshow(frame_rgb)
        inset_ax.axis('off')  # Hide axes for the frame

    # Fix the Y limits to make everything fit visually
    plt.ylim(0, 1.05)

    # Add labels and legend
    plt.xlabel("Time (frames)", fontsize=14)
    plt.ylabel("Anomaly Score", fontsize=14)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True)

    # Save and show
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)  # Adjust layout to avoid warnings
    plt.savefig(output_path.replace(".mp4", "_graph.png"))
    plt.show()


if __name__ == "__main__":
    video_path = r"C:\Users\tatra\Downloads\The Most UNEXPECTED Finishes in UFC History 😱.mp4"
    model_path = "saved_models/84.98/"
    gt_path = r"D:\MAVD2\list\gt.npy"
    output_path = r"c:\Users\tatra\Downloads\output_video.mp4"

    # Perform inference
    predictions, frames = infer_on_video(video_path, model_path, gt_path)

    # Plot predictions with graph
    frame_skip = 16
    plot_predictions_with_graph(frames, predictions, frame_skip, output_path)
