import torch
import numpy as np
from pytorchfi.util import random_value, relative_iou, setup_dicts, compute_iou, compute_mAP, compute_ratio


f_ten = [{'boxes': torch.tensor([[116.3984, 111.6682, 225.8636, 221.7298],
        [196.7261, 193.1804, 267.8563, 260.9183],
        [198.2962, 186.6753, 269.3496, 259.2556],
        [113.3028, 213.8261, 145.1692, 255.1516],
        [195.7266, 186.6376, 270.7689, 258.0387],
        [ 71.2014, 543.2361, 404.0000, 619.4427]], device='cpu'), 'labels': torch.tensor([13,  3,  8,  3,  6, 15], device='cpu'), 'scores': torch.tensor([0.9992, 0.7143, 0.5782, 0.4528, 0.2201, 0.0563], device='cpu')}]

g_ten = [{'boxes': torch.tensor([[116.4000, 111.7068, 225.8904, 221.7406],
        [196.7495, 193.0036, 267.8851, 260.9280],
        [198.3679, 186.7295, 269.3483, 259.2439],
        [113.3215, 213.8959, 145.1490, 255.1735],
        [195.7583, 186.6151, 270.7934, 258.0372],
        [ 71.0886, 543.2857, 404.0000, 619.2117]], device='cpu'), 'labels': torch.tensor([13,  3,  8,  3,  6, 15], device='cpu'), 'scores': torch.tensor([0.9992, 0.7310, 0.5835, 0.4523, 0.2080, 0.0530], device='cpu')}]


for pred_g, pred_f in zip(g_ten, f_ten):
    # sorted_f_bb = np.empty(pred_g['boxes'].shape[0])
    # sorted_f_lab = np.empty(pred_g['labels'].shape[0])
    # sorted_f_conf = np.empty(pred_g['scores'].shape[0])

    fault_bbs = pred_f['boxes']
    fault_labs = pred_f['labels']
    fault_confs = pred_f['scores']

    critical = 0
    SDC = 0
    masked = 0

    for bb_idx in range(len(pred_g['boxes'])):
        bb = pred_g['boxes'][bb_idx]
        faulty_disatnces1 = np.linalg.norm(bb[0:2] - fault_bbs[:,0:2], axis=1, ord=2)
        faulty_disatnces2 = np.linalg.norm(bb[2:4] - fault_bbs[:,2:4], axis=1, ord=2)
        # print(faulty_disatnces1)
        # print(faulty_disatnces2)
        f_buffer = faulty_disatnces1 + faulty_disatnces2

        # take the lowest one
        f_candidate_idx = np.argmin(f_buffer)

        # take the array correspinding to the lowest distance from the reference gt_bb 
        f_candidate_bb = fault_bbs[f_candidate_idx]
        f_candidate_lab = fault_labs[f_candidate_idx]
        f_candidate_conf = fault_confs[f_candidate_idx]

        f_score = compute_iou(bb, f_candidate_bb)
        ratio = compute_ratio(bb, f_candidate_bb)

        if f_score  ==  1:

            if f_candidate_lab == pred_g['labels'][bb_idx]:
                if f_candidate_conf / pred_g['scores'][bb_idx] > 0.5:
                    masked += 1
                else: 
                    SDC += 1
            else:
                print(f_candidate_lab)
                print(pred_g['labels'][bb_idx])
                print('label diverse e iou alto')
                critical += 1

        elif f_score < 1 and f_score > 0.6:

            if f_candidate_lab == pred_g['labels'][bb_idx]:
                if f_candidate_conf / pred_g['scores'][bb_idx] > 0.5:
                    SDC += 1
                else: 
                    print('confidenza bassa')
                    critical += 1
            else:
                print('label diverse e iou medio')
                critical += 1

        elif f_score < 0.6:
            # print(f_score)
            print('iou basso')
            critical += 1
            

        fault_bbs = np.delete(fault_bbs, f_candidate_idx, axis = 0)
        fault_labs = np.delete(fault_labs, f_candidate_idx, axis = 0)
        fault_confs = np.delete(fault_confs, f_candidate_idx, axis = 0)

        if len(fault_bbs) == 0:
            break


print(masked)