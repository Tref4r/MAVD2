import torch
import torch.nn as nn
import torch.nn.functional as F


def CosLoss(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].reshape(a[item].shape[0],-1),
                                      b[item].reshape(b[item].shape[0],-1)))
    return loss

def match_sparsify(V, A_, F_, P_):
    
    batch_size = V.shape[0]
    time = V.shape[1]
    max_features = V.shape[-1]
    
    def pad_tensor(V, O, b, t, m):
        O_m = O.shape[-1]
        O_flat = O.view(b * t, O_m)  
        V_flat = V.view(b * t, m)  

        O_flat_norm = F.normalize(O_flat, p=2, dim=0)
        V_flat_norm = F.normalize(V_flat, p=2, dim=0)      

        similarity = torch.mm(O_flat_norm.t(), V_flat_norm)

        _, S = torch.topk(similarity, m, dim=1)

        I = torch.empty(O_m, dtype=torch.long)  
        used_values = set()  

        for i in range(O_m):
            value_found = False
            for j in range(m):  
                candidate = S[i, j].item() 
                if candidate not in used_values:  
                    I[i] = candidate  
                    used_values.add(candidate)  
                    value_found = True
                    break  
            if not value_found:
                print(f"Warning: No unique value found for index {i}.")

        extended_i = torch.empty(m, dtype=torch.long)
        extended_i[:O_m] = I 

        remaining_values = [x for x in range(m) if x not in used_values]
        extended_i[O_m:] = torch.tensor(remaining_values[:m-O_m])  

        padded_O = F.pad(O, (0, m-O_m))
        padded_O = padded_O[:, :, extended_i]

        return padded_O
    
    padded_A = pad_tensor(V, A_, batch_size, time ,max_features)
    padded_F = pad_tensor(V, F_, batch_size, time ,max_features)
    padded_P = pad_tensor(V, P_, batch_size, time ,max_features)
    
    return V, padded_A, padded_F, padded_P

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def get_loss(self, result, label):

        output = result['output']
        output_loss = self.bce(output, label)

        return output_loss
        
    def forward(self, v_result, a_result, f_result, p_result, label):

        label = label.float()

        v_loss = self.get_loss(v_result, label)
        a_loss = self.get_loss(a_result, label)
        f_loss = self.get_loss(f_result, label)
        p_loss = self.get_loss(p_result, label)

        U_MIL_loss = v_loss + a_loss + f_loss + p_loss

        loss_dict = {}
        loss_dict['U_MIL_loss'] = U_MIL_loss

        return U_MIL_loss, loss_dict
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class DISL_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.contrastive_loss = ContrastiveLoss()  # Replace CrossEntropyLoss with ContrastiveLoss

    def norm(self, data):
        l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
        return torch.div(data, l2)
    
    def get_seq_matrix(self, seq_len):
        N = seq_len.size(0)
        M = seq_len.max().item()
        seq_matrix = torch.zeros((N, M))
        for j, val in enumerate(seq_len):
            seq_matrix[j, :val] = 1
        seq_matrix = seq_matrix.cuda()
        return seq_matrix

    def get_mil_loss(self, result, label):

        output = result['output']
        output_loss = self.bce(output, label)

        return output_loss
    
    def contrastive_loss_fn(self, q, p, label):
        return self.contrastive_loss(q, p, label)

    def get_alignment_loss(self, v_result, va_result, vf_result, vp_result, seq_len):
        def distance(x, y):
            return CosLoss(x, y)
        
        V = v_result["satt_f"].detach().clone()
        A = va_result["satt_f"]
        F = vf_result["satt_f"]
        P = vp_result["satt_f"]

        batch_size = V.shape[0]

        V, A, F, P = match_sparsify(V, A, F, P)

        d_VA = distance(V, A)/batch_size
        d_VF = distance(V, F)/batch_size
        d_VP = distance(V, P)/batch_size
        d_AP = distance(A, P)/batch_size
        d_AF = distance(A, F)/batch_size
        d_FP = distance(F, P)/batch_size

        seq_matrix = self.get_seq_matrix(seq_len)
        V = v_result["avf_out"].detach().clone() * seq_matrix
        A = va_result["avf_out"] * seq_matrix
        F = vf_result["avf_out"] * seq_matrix
        P = vp_result["avf_out"] * seq_matrix

        ce_VA = self.contrastive_loss_fn(V, A, torch.ones_like(V[:, 0]))
        ce_VF = self.contrastive_loss_fn(V, F, torch.ones_like(V[:, 0]))
        ce_VP = self.contrastive_loss_fn(V, P, torch.ones_like(V[:, 0]))
        ce_AF = self.contrastive_loss_fn(A, F, torch.ones_like(A[:, 0]))
        ce_AP = self.contrastive_loss_fn(A, P, torch.ones_like(A[:, 0]))
        ce_FP = self.contrastive_loss_fn(F, P, torch.ones_like(F[:, 0]))

        return d_VA + d_VF + d_VP + d_AP + d_AF + d_FP + ce_VA + ce_VF + ce_VP + ce_AF + ce_AP + ce_FP

    def get_contrastive_loss(self, vafp_result, label):
        logits = vafp_result["avf_out"]
        
        # Ensure label has the same number of dimensions as logits
        if label.dim() == 1:
            label = label.unsqueeze(1)  # Add a dimension to make it 2D

        # Ensure dimensions match
        if logits.shape[1] != label.shape[1]:
            min_dim = min(logits.shape[1], label.shape[1])
            logits = logits[:, :min_dim]  # Slice logits to match label dimensions
            label = label[:, :min_dim]  # Slice label to match logits dimensions

        return self.contrastive_loss(logits, label, torch.ones_like(label))  # Use ContrastiveLoss
     
    def forward(self, v_result, va_result, vf_result, vp_result, vafp_result, label, seq_len, lamda1, lamda2, lamda3):

        label = label.float()
        a_loss = self.get_mil_loss(va_result, label)
        f_loss = self.get_mil_loss(vf_result, label)
        p_loss = self.get_mil_loss(vp_result, label)
        rafp_loss = self.get_mil_loss(vafp_result, label)

        ma_loss = self.get_alignment_loss(v_result, va_result, vf_result, vp_result, seq_len)
        ma_loss = ma_loss + 0.01*(a_loss + f_loss + p_loss)
        
        contrastive_loss = self.get_contrastive_loss(vafp_result, label)  # Replace CrossEntropyLoss

        total_loss = lamda1*ma_loss + lamda2*rafp_loss + lamda3*contrastive_loss

        loss_dict = {}
        loss_dict['MA_loss'] = ma_loss
        loss_dict['M_MIL_loss'] = rafp_loss
        loss_dict['Contrastive_loss'] = contrastive_loss  # Update key

        return total_loss, loss_dict