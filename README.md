# MAVD
  Multi-Modal Fusion Techniques for Abnormal Events
  Detection in Videos

<p align="center">
    <img src=pipeline_violence.png width="800" height="300"/>
</p>

[Paper Link](https://arxiv.org/abs/2501.07496) | [Code](https://github.com/Tref4r/MAVD2/tree/release/1.0)

## Abstract
With the rapid increase in video data and the growing need for security and public safety, detecting abnormal
events in videos has become increasingly important. In recent years, deep learning techniques have been
extensively applied to improve automatic anomaly detection systems. Abnormal events often manifest through
multiple modalities such as appearance, motion, audio signals, and human pose, highlighting the need for
effective multimodal feature extraction and integration. However, existing fusion mechanisms often struggle
with aligning heterogeneous features and fully leveraging cross-modal information. We experimented with
several commonly used multimodal fusion techniques and applied an adaptive framework that integrates
RGB, optical flow, audio, and pose modalities while enhancing semantic consistency across modalities. Our
approach dynamically regulates the contribution of each modality based on contextual relevance to improve
anomaly detection performance.


## Requirements

### System Requirements
- Docker Engine
- NVIDIA Docker toolkit
- NVIDIA GPU with CUDA support
- At least 8GB of GPU memory
- At least 4GB of system memory

Our Docker images are built with:
- CUDA 11.8 with cuDNN 8 (fusion service)
- CUDA 11.3 with cuDNN 8 (mmaction service)
- PyTorch 2.2.1 (fusion service)
- PyTorch 1.12.1 (mmaction service)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Tref4r/MAVD2.git
cd MAVD2
git checkout release/1.0

# Start the services
docker compose -f Compose/compose.yml up --build -d
```

The system consists of two Docker services:
1. `fusion`: Main violence detection service
2. `mmaction`: Pose extraction service

## Project Structure

```
├── Compose/                # Docker compose configuration
│   └── compose.yml        # Services orchestration
│
├── fusion/                 # Violence Detection Component
│   ├── model/             # Core model implementations
│   │   ├── multimodal.py  # Multimodal fusion architecture
│   │   ├── unimodal.py    # Single modality networks
│   │   ├── projection.py  # Feature projection modules
│   │   ├── tcn.py        # Temporal modeling
│   │   └── self_attention.py  # Attention mechanisms
│   ├── dataset/           # Dataset handling
│   │   └── dataset_loader.py
│   ├── config/            # Configuration files
│   │   ├── options.py     # Training options
│   │   └── options_test.py # Testing options
│   ├── losses/            # Loss functions
│   ├── utils/             # Utility functions
│   ├── Docker/            # Docker configuration
│   ├── Model/             # Pre-trained models
│   │   ├── Model_extract/ # Feature extraction models
│   │   │   ├── flow_imagenet.pt
│   │   │   ├── i3d_rgb_imagenet.pt
│   │   │   └── vggish-10086976.pth
│   │   └── Model_Infer/   # Inference models
│   │       ├── v_model.pth    # RGB model
│   │       ├── a_model.pth    # Audio model
│   │       ├── f_model.pth    # Flow model
│   │       ├── p_model.pth    # Pose model
│   │       └── *_model.pth    # Combined models
│   ├── list/              # Data lists
│   │   ├── video_train.list
│   │   ├── video_test.list
│   │   └── gt.npy         # Ground truth labels
│   └── train_test/        # Training and testing scripts
│       ├── train.py
│       └── test.py
│
└── mmaction/              # Pose Extraction Component
    ├── mmaction/          # Core MMAction library
    │   ├── apis/          # APIs
    │   ├── models/        # Model implementations
    │   ├── datasets/      # Dataset processing
    │   └── utils/         # Utilities
    ├── configs/           # Model configurations
    │   ├── _base_/        # Base configs
    │   └── skeleton/      # Pose-related configs
    ├── model/             # Pre-trained models
    │   ├── det/           # Detection models
    │   └── pose/          # Pose estimation models
    └── Docker/            # Docker configuration
```

Key Files:
- `get_models.py`: Main training script
- `get_result.py`: Evaluation script
- `video_inference.py`: Single video inference
- `video_inference_visualization.py`: Results visualization
- `extract_feats_pose_api.py`: Pose feature extraction API
- `main.py`: MMAction service entry point

## Usage

### Training

Training is done within the fusion container:

```bash
# Enter the fusion container
docker exec -it fusion_container bash

# Run training
python get_models.py
```

### Testing

Testing can be performed inside the fusion container:

```bash
# Enter the fusion container
docker exec -it fusion_container bash

# Run evaluation
python get_result.py
```

### Single Video Inference

For running inference on a single video:

```bash
# Enter the fusion container
docker exec -it fusion_container bash

# Run inference
python video_inference.py
```

### Viewing Results

Results and model outputs are saved in the mounted volumes:
- Model checkpoints: `fusion/Model/`
- Pose features: `fusion/pose_feats/`
- Evaluation results: Will be displayed in the terminal output


## Citation

If you find this repo useful for your research, please consider citing our paper:
```
@unpublished{tung2025multi,
  title={Multi-Modal Fusion Techniques for Abnormal Events Detection in Videos},
  author={Ta Tran Thanh Tung, Dinh Cong Binh, Vu Minh Hoang, Tran Minh Hoan, Vuong Tu Binh},
  year={2025}
}
```

## Acknowledgements

This project builds upon several excellent works:
- [MAVD](https://github.com/xjpp2016/MAVD)
- [mmaction2](https://github.com/open-mmlab/mmaction2)

We sincerely thank the authors of these projects for their valuable contributions.