import numpy as np
import torch
import torch.nn.functional as F
import argparse

def interpolate_npy(input_path, target_shape):
    data = np.load(input_path)
    
    if data.shape[0] == target_shape:
        return data.reshape(target_shape, 512)

    data_tensor = torch.from_numpy(data).squeeze().unsqueeze(0).unsqueeze(0)
    
    new_tensor = F.interpolate(data_tensor, 
                             size=(target_shape, 512),
                             mode='bilinear', 
                             align_corners=False)
    
    new_data = new_tensor.squeeze(0).squeeze(0).numpy()
    
    return new_data

