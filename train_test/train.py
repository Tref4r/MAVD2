import torch
from tqdm import tqdm

def train(v_net, a_net, f_net, va_net, vf_net, vaf_net, dataloader, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3):

    with torch.set_grad_enabled(True):

        f_v, f_a, f_f, label  = next(dataloader)

        v_net.train()
        a_net.train()
        f_net.train()
        va_net.train()
        vf_net.train()
        vaf_net.train()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]
        f_a = f_a[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]

        v_data = f_v.cuda()
        a_data = f_a.cuda()
        f_data = f_f.cuda()
        label = label.cuda()

        v_predict = v_net(v_data, seq_len)
        a_predict = a_net(a_data, seq_len)
        f_predict = f_net(f_data, seq_len)

        total_loss, loss_dict_list = criterion(v_predict, a_predict, f_predict, label)

        v_input = v_predict["satt_f"].detach().clone()
        a_input = a_predict["satt_f"].detach().clone()
        f_input = f_predict["satt_f"].detach().clone()
        a_input = va_net(a_input)
        f_input = vf_net(f_input)

        vaf_input = torch.cat([v_input, a_input, f_input], dim=-1)

        va_output = a_net(a_input, seq_len, em_flag=False)
        vf_output = f_net(f_input, seq_len, em_flag=False)
        vaf_output = vaf_net(vaf_input, seq_len)

        total_loss_disl, loss_dict_list_disl = criterion_disl(v_predict, va_output, vf_output, vaf_output, label, seq_len.cuda(), lamda1, lamda2, lamda3)

        # Dynamic loss weighting
        lambda1 = 1.0 / (loss_dict_list["U_MIL_loss"] + 1e-8)
        lambda2 = 1.0 / (loss_dict_list_disl["MA_loss"] + 1e-8)
        lambda3 = 1.0 / (loss_dict_list_disl["M_MIL_loss"] + 1e-8)

        total_loss = lambda1 * loss_dict_list["U_MIL_loss"] + \
                     lambda2 * loss_dict_list_disl["MA_loss"] + \
                     lambda3 * loss_dict_list_disl["M_MIL_loss"] + \
                     loss_dict_list_disl["Triplet_loss"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return loss_dict_list, loss_dict_list_disl



