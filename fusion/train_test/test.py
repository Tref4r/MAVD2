import torch
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def test(v_net, a_net, f_net, p_net, va_net, vf_net, vp_net, vafp_net, test_loader, gt, test_info, epoch, max_seqlen):
    
    with torch.no_grad():

        v_net.eval()
        a_net.eval()
        f_net.eval()
        p_net.eval()
        va_net.eval()
        vf_net.eval()
        vp_net.eval()
        vafp_net.eval()
        
        m_pred = torch.zeros(0).cuda()

        for i, (f_v, f_a, f_f, f_p) in tqdm(enumerate(test_loader)):
            
            v_data = f_v.cuda()
            a_data = f_a.cuda()
            f_data = f_f.cuda()
            p_data = f_p.cuda()

            v_res = v_net(v_data)
            a_res = a_net(a_data)
            f_res = f_net(f_data)
            p_res = p_net(p_data)

            mix_f = torch.cat([v_res["satt_f"], va_net(a_res["satt_f"]), vf_net(f_res["satt_f"]), vp_net(p_res["satt_f"])], dim=-1)
            m_out = vafp_net(mix_f)
            
            m_out = torch.mean(m_out["output"], 0)
            m_pred = torch.cat((m_pred, m_out))

        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)

        