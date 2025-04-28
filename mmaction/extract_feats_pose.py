import argparse
import tempfile
import warnings

import cv2
import mmengine
import numpy as np
import torch

from mmaction.apis import (detection_inference, inference_recognizer,
                           inference_skeleton, init_recognizer, pose_inference)
from mmaction.utils import frame_extract

import json
from pathlib import Path
import os
from datetime import datetime

try:
    from mmdet.apis import init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `init_detector` form `mmdet.apis`. '
                  'These apis are required in skeleton-based applications! ')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]



def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou


def skeleton_based_action_recognition(args, pose_results, h, w):
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset

    skeleton_model = init_recognizer(
        skeleton_config, args.skeleton_checkpoint, device='cpu')
    result = inference_skeleton(skeleton_model, pose_results, (h, w))
    action_idx = result.pred_score.argmax().item()
    return label_map[action_idx]


def skeleton_based_stdet(args, label_map, human_detections, pose_results,
                         num_frame, clip_len, frame_interval, h, w):
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config.model.cls_head.num_classes = num_class
    skeleton_stdet_model = init_recognizer(skeleton_config,
                                           args.skeleton_stdet_checkpoint,
                                           args.device)

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        num_frame = len(frame_inds)  # 30

        pose_result = [pose_results[ind] for ind in frame_inds]

        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person
            skeleton_prediction.append([])

            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, num_frame, num_keypoint))  # M T V

            # pose matching
            person_bbox = proposal[i][:4]
            area = expand_bbox(person_bbox, h, w)

            for j, poses in enumerate(pose_result):  # num_frame
                max_iou = float('-inf')
                index = -1
                if len(poses['keypoints']) == 0:
                    continue
                for k, bbox in enumerate(poses['bboxes']):
                    iou = cal_iou(bbox, area)
                    if max_iou < iou:
                        index = k
                        max_iou = iou
                keypoint[0, j] = poses['keypoints'][index]
                keypoint_score[0, j] = poses['keypoint_scores'][index]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            output = inference_recognizer(skeleton_stdet_model, fake_anno)
            # for multi-label recognition
            score = output.pred_score.tolist()
            for k in range(len(score)):  # 81
                if k not in label_map:
                    continue
                if score[k] > args.action_score_thr:
                    skeleton_prediction[i].append((label_map[k], score[k]))

        skeleton_predictions.append(skeleton_prediction)
        prog_bar.update()

    return timestamps, skeleton_predictions

def find_mp4_files(directory):
    # Check if directory is actually a file
    if os.path.isfile(directory) and directory.lower().endswith('.mp4'):
        return [Path(directory)]
    # If it's a directory, search for MP4 files recursively    
    return list(Path(directory).rglob("*.mp4"))

def main():
    
    with open("config.json", "r") as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)

    folder_data = args.input_data
    # Verify if the path exists
    if not os.path.exists(folder_data):
        raise ValueError(f"The path {folder_data} does not exist")
        
    list_mp4 = find_mp4_files(directory=folder_data)
    if not list_mp4:
        raise ValueError(f"No MP4 files found in {folder_data}")

    for idx, mp4 in enumerate(list_mp4):
        start_time = datetime.now() 
        tmp_dir = tempfile.TemporaryDirectory()
        print(f'{idx}/{len(list_mp4)} --- Current ---> {mp4}')
        os.makedirs('Extract_feats/final', exist_ok=True)
        name_mp4 = os.path.split(mp4)[-1]
        name_mp4 = name_mp4.replace('.mp4','__pose.npy')
        if name_mp4 in os.listdir(os.path.join('Extract_feats','final')):
            print(f'Existed --- {name_mp4}')
            continue
        args.video = mp4
        frame_paths, original_frames = frame_extract(
            args.video, out_dir=tmp_dir.name)
        num_frame = len(frame_paths)
        h, w, _ = original_frames[0].shape

        # Get Human detection results and pose results
        human_detections, _ = detection_inference(
            args.det_config,
            args.det_checkpoint,
            frame_paths,
            args.det_score_thr,
            device=args.device)
        torch.cuda.empty_cache()
        pose_results, pose_datasample = pose_inference(
            args.pose_config,
            args.pose_checkpoint,
            frame_paths,
            human_detections,
            device=args.device)
        
        print('Use skeleton-based recognition')
        action_result = skeleton_based_action_recognition(
            args, pose_results, h, w)
    
        os.rename(os.path.join('Extract_feats','raw_feats.npy'),f'Extract_feats/final/{name_mp4}')

        tmp_dir.cleanup()
        end_time = datetime.now()  
        elapsed_time = end_time - start_time  
        print(f"Time: {elapsed_time}")

if __name__ == '__main__':
    main()