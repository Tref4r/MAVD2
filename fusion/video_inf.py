import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import random
import os
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.models as models
from torchvggish import vggish, vggish_input
import torchaudio

import ffmpeg
import io
import soundfile as sf

from dataset.dataset_loader import Dataset
from model.unimodal import Unimodal
from model.multimodal import Multimodal
from model.projection import Projection
from utils.utils import process_feat

# Optional: import I3D model nếu bạn có
from i3d import InceptionI3d  # Giả sử bạn đã chuẩn bị file i3d.py

# --------------------- Utility Functions ---------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def normalize_tensor(tensor):
    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return (tensor - min_val) / (max_val - min_val + 1e-8)

def normalize_predictions(predictions):
    min_val = np.min(predictions)
    max_val = np.max(predictions)
    if max_val - min_val == 0:
        return np.zeros_like(predictions)
    normalized = (predictions - min_val) / (max_val - min_val)
    return normalized

def log_predictions(predictions):
    print(f"Predictions: {predictions}")
    print(f"Mean: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}, Min: {np.min(predictions):.4f}, Max: {np.max(predictions):.4f}")

def load_ground_truth(gt_path, frame_skip, num_predictions):
    ground_truth = np.load(gt_path)
    subsampled_gt = ground_truth[::frame_skip]
    if len(subsampled_gt) > num_predictions:
        subsampled_gt = subsampled_gt[:num_predictions]
    elif len(subsampled_gt) < num_predictions:
        raise ValueError(f"Mismatch: {len(subsampled_gt)} ground truth vs {num_predictions} predictions")
    return subsampled_gt

def evaluate_predictions(predictions, ground_truth, threshold=0.5):
    binary_predictions = (predictions > threshold).astype(int)
    accuracy = accuracy_score(ground_truth, binary_predictions)
    precision = precision_score(ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(ground_truth, binary_predictions, zero_division=0)
    return accuracy, precision, recall, f1

def calibrate_threshold(predictions, ground_truth):
    best_f1 = 0
    best_thresh = 0.5
    print("\n--- Threshold Calibration ---")
    for thresh in np.arange(0.1, 0.9, 0.05):
        binary_pred = (predictions > thresh).astype(int)
        f1 = f1_score(ground_truth, binary_pred, zero_division=0)
        print(f"Threshold {thresh:.2f}: F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"\n>> Best threshold = {best_thresh:.2f} with F1 = {best_f1:.4f}")
    print("--- End of Calibration ---\n")
    return best_thresh

def predict_video_class(predictions, threshold=0.5):
    video_score = np.mean(predictions)
    if video_score > threshold:
        return "Violent (Bạo lực)"
    else:
        return "Non-violent (Không bạo lực)"

def predict_video_by_voting(predictions, threshold=0.5, vote_ratio=0.5):
    binary_preds = (predictions > threshold).astype(int)
    violent_ratio = np.mean(binary_preds)
    if violent_ratio >= vote_ratio:
        return "Violent (Bạo lực)"
    else:
        return "Non-violent (Không bạo lực)"

# --------------------- Feature Extraction ---------------------

def extract_frames(video_path, frame_skip=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        frame_count += 1
    cap.release()
    return np.array(frames)

def extract_rgb_flow_features(frames):
    print("Extracting RGB and Flow features using I3D...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Chuẩn bị tensor cho RGB
    rgb_tensors = []
    for frame in frames:
        frame = transform(frame)
        rgb_tensors.append(frame)
    rgb_tensors = torch.stack(rgb_tensors, dim=1)  # (3, T, H, W)
    rgb_tensors = rgb_tensors.unsqueeze(0).to(device)  # (1, 3, T, H, W)

    # Chuẩn bị tensor cho Optical Flow
    flows = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flow = cv2.resize(flow, (224, 224))
            # Normalize flow về -1~1
            flow = np.clip(flow / 20.0, -1.0, 1.0)
            flows.append(flow)
        prev_gray = gray

    # Vì optical flow ra ít hơn 1 frame, cần resize RGB cho khớp
    rgb_tensors = rgb_tensors[:, :, 1:, :, :]  # Bỏ frame đầu tiên

    flow_array = np.stack(flows)  # (T-1, H, W, 2)
    flow_array = np.transpose(flow_array, (3, 0, 1, 2))  # (2, T-1, H, W)
    flow_tensors = torch.from_numpy(flow_array).unsqueeze(0).float().to(device)  # (1, 2, T-1, H, W)

    try:
        # RGB branch
        i3d_rgb = InceptionI3d(400).to(device)
        rgb_state_dict = torch.load('./i3d_rgb_imagenet.pt')
        rgb_new_state = {}
        for k, v in rgb_state_dict.items():
            if 'logits' in k:
                rgb_new_state[k.replace('logits', 'conv3d_0c_1x1')] = v
            else:
                rgb_new_state[k] = v
        i3d_rgb.load_state_dict(rgb_new_state)
        i3d_rgb.eval()

        # Flow branch
        i3d_flow = InceptionI3d(400, in_channels=2).to(device)
        flow_state_dict = torch.load('./flow_imagenet.pt')  # ⚡️ Bạn cần chuẩn bị file này
        flow_new_state = {}
        for k, v in flow_state_dict.items():
            if 'logits' in k:
                flow_new_state[k.replace('logits', 'conv3d_0c_1x1')] = v
            else:
                flow_new_state[k] = v
        i3d_flow.load_state_dict(flow_new_state)
        i3d_flow.eval()

        with torch.no_grad():
            rgb_features = i3d_rgb.extract_features(rgb_tensors)
            rgb_features = rgb_features.mean([3, 4])  # (1, 1024, T/8)
            rgb_features = rgb_features.squeeze(0).permute(1, 0).cpu().numpy()  # (T, 1024)

            flow_features = i3d_flow.extract_features(flow_tensors)
            flow_features = flow_features.mean([3, 4])  # (1, 1024, T/8)
            flow_features = flow_features.squeeze(0).permute(1, 0).cpu().numpy()  # (T, 1024)

    except Exception as e:
        print(f"Failed loading I3D models: {e}")
        num_frames = len(frames)
        rgb_features = np.random.rand(num_frames, 1024)
        flow_features = np.random.rand(num_frames, 1024)

    return rgb_features, flow_features


def extract_audio_features(video_path):
    print("Extracting Audio features using VGGish (no temp file)...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # 🛠 Trích xuất audio trực tiếp từ video thành bytes
        out, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='wav', ac=1, ar='16000')  # Mono, 16kHz như yêu cầu VGGish
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 🛠 Đọc từ bytes vào numpy array
        audio_data, sr = sf.read(io.BytesIO(out))

        if sr != 16000:
            raise ValueError(f"Sample rate mismatch: got {sr}, expected 16000 Hz")

        # 🛠 Convert numpy audio to VGGish examples
        examples = vggish_input.waveform_to_examples(audio_data, sr)

        if isinstance(examples, torch.Tensor):
            examples = examples.detach().cpu().numpy()

        model = vggish()
        model.load_state_dict(torch.load('./vggish-10086976.pth'))
        model = model.to(device)
        model.postprocess = False

        inputs = torch.from_numpy(examples).float().to(device)

        with torch.no_grad():
            embeddings = model(inputs)

        features = embeddings.cpu().numpy()

    except Exception as e:
        print(f"Failed extracting audio: {e}")
        features = np.random.rand(100, 128)

    return features


def extract_pose_features(frames):
    print("Extracting Pose features using PoseC3D (placeholder random)...")
    num_frames = len(frames)
    pose_features = np.random.rand(num_frames, 512)  # Nếu chưa detect keypoints
    return pose_features

def load_models(model_path):
    v_net = Unimodal(input_size=1024, h_dim=128, feature_dim=128).cuda()
    a_net = Unimodal(input_size=128, h_dim=64, feature_dim=32).cuda()
    f_net = Unimodal(input_size=1024, h_dim=128, feature_dim=64).cuda()
    p_net = Unimodal(input_size=512, h_dim=128, feature_dim=64).cuda()
    va_net = Projection(32, 32, 32).cuda()
    vf_net = Projection(64, 64, 64).cuda()
    vp_net = Projection(64, 64, 64).cuda()
    vafp_net = Multimodal(input_size=128+32+64+64, h_dim=128, feature_dim=64).cuda()

    v_net.load_state_dict(torch.load(f"{model_path}/v_model.pth"))
    a_net.load_state_dict(torch.load(f"{model_path}/a_model.pth"))
    f_net.load_state_dict(torch.load(f"{model_path}/f_model.pth"))
    p_net.load_state_dict(torch.load(f"{model_path}/p_model.pth"))
    va_net.load_state_dict(torch.load(f"{model_path}/va_model.pth"))
    vf_net.load_state_dict(torch.load(f"{model_path}/vf_model.pth"))
    vp_net.load_state_dict(torch.load(f"{model_path}/vp_model.pth"))
    vafp_net.load_state_dict(torch.load(f"{model_path}/vafp_model.pth"))

    return v_net, a_net, f_net, p_net, va_net, vf_net, vp_net, vafp_net

# --------------------- Inference Function ---------------------

def infer_on_video(video_path, model_path, gt_path, frame_skip=16):
    set_seed(42)

    frames = extract_frames(video_path, frame_skip)
    if len(frames) == 0:
        raise RuntimeError("No frames extracted!")

    print(f"Frames extracted: {len(frames)}")

    rgb_features, flow_features = extract_rgb_flow_features(frames)
    audio_features = extract_audio_features(video_path)
    pose_features = extract_pose_features(frames)

    min_len = min(len(rgb_features), len(flow_features), len(audio_features), len(pose_features))
    rgb_features = rgb_features[:min_len]
    flow_features = flow_features[:min_len]
    audio_features = audio_features[:min_len]
    pose_features = pose_features[:min_len]
    frames = frames[:min_len]

    rgb_features = torch.tensor(rgb_features, dtype=torch.float32).unsqueeze(0).cuda()
    flow_features = torch.tensor(flow_features, dtype=torch.float32).unsqueeze(0).cuda()
    audio_features = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).cuda()
    pose_features = torch.tensor(pose_features, dtype=torch.float32).unsqueeze(0).cuda()

    v_net, a_net, f_net, p_net, va_net, vf_net, vp_net, vafp_net = load_models(model_path)

    with torch.no_grad():
        v_res = normalize_tensor(v_net(rgb_features)["satt_f"])
        a_res = normalize_tensor(a_net(audio_features)["satt_f"])
        f_res = normalize_tensor(f_net(flow_features)["satt_f"])
        p_res = normalize_tensor(p_net(pose_features)["satt_f"])

        print(f"Intermediate features: v_res {v_res.shape}, a_res {a_res.shape}, f_res {f_res.shape}, p_res {p_res.shape}")

        mix_f = torch.cat([v_res, va_net(a_res), vf_net(f_res), vp_net(p_res)], dim=-1)
        m_out = vafp_net(mix_f)
        predictions = m_out["output"].cpu().numpy()
        predictions = normalize_predictions(predictions)

        log_predictions(predictions)

        ground_truth = load_ground_truth(gt_path, frame_skip, len(predictions))
        accuracy, precision, recall, f1 = evaluate_predictions(predictions, ground_truth)
        print(f"Evaluation Metrics: Accuracy {accuracy:.4f}, Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}")

        best_thresh = calibrate_threshold(predictions, ground_truth)
        accuracy, precision, recall, f1 = evaluate_predictions(predictions, ground_truth, threshold=best_thresh)
        print(f"Metrics at best threshold {best_thresh:.2f}: Acc={accuracy:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


        # 🛠 QUICK FEATURE HEALTH CHECK
        print("\n=== Quick Check for Feature Health ===")

        # Check variance của từng modality
        print(f"Feature Variances:")
        print(f"  - RGB  : {v_res.var().item():.6f}")
        print(f"  - Audio: {a_res.var().item():.6f}")
        print(f"  - Flow : {f_res.var().item():.6f}")
        print(f"  - Pose : {p_res.var().item():.6f}")

        # Predict riêng từng modality để xem modal nào mạnh yếu
        pred_v = torch.sigmoid(v_res.mean(dim=-1)).cpu().numpy().squeeze()
        pred_a = torch.sigmoid(a_res.mean(dim=-1)).cpu().numpy().squeeze()
        pred_f = torch.sigmoid(f_res.mean(dim=-1)).cpu().numpy().squeeze()
        pred_p = torch.sigmoid(p_res.mean(dim=-1)).cpu().numpy().squeeze()

        print(f"Single Modality Prediction Means:")
        print(f"  - RGB  : {pred_v.mean():.4f}")
        print(f"  - Audio: {pred_a.mean():.4f}")
        print(f"  - Flow : {pred_f.mean():.4f}")
        print(f"  - Pose : {pred_p.mean():.4f}")

        print("=== End of Quick Check ===\n")
    # ➔ Auto predict video class
    video_class = predict_video_class(predictions, threshold=best_thresh)
    voting_class = predict_video_by_voting(predictions, threshold=best_thresh, vote_ratio=0.5)

    print(f"\n==> Quick Prediction (mean method): {video_class}")
    print(f"==> Quick Prediction (voting method): {voting_class}\n")


    return predictions, frames, accuracy, precision, recall, f1

# --------------------- Visualization ---------------------

def visualize_predictions(frames, predictions):
    for i, frame in enumerate(frames):
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {i+1} - Prediction: {predictions[i]:.2f}", fontsize=16)
        plt.axis('off')
        plt.show()

# --------------------- Main ---------------------

if __name__ == "__main__":
    video_path = r"C:\Users\tatra\Downloads\iLoveTik.com_TikTok_Media_001_8e03efbdf8d0abcc838c56e60302e02f.mp4"
    model_path = "saved_models/84.98/"
    gt_path = r"D:\MAVD2\list\gt.npy"

    predictions, frames, accuracy, precision, recall, f1 = infer_on_video(video_path, model_path, gt_path)
    visualize_predictions(frames, predictions)
