from typing import *
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from collections import defaultdict
import numpy as np
from copy import deepcopy
import math

# save true bounding boxes corresponding to each label
# gt_dict = defaultdict(lambda:[])
# gt_labels = torch.squeeze(gt_labels)
# for idx, label in enumerate(gt_labels.tolist()):
#     gt_bb = torch.squeeze(_gt_bbs)
#     gt_dict[str(label)].append(gt_bb[idx].numpy())

# # save predicted bounding boxes corresponding to each label
# pred_dict = defaultdict(lambda:[])
# for (idx1, label_pred), score, (idx2, label_gt) in zip(enumerate(pred_labels.tolist()), pred_scores.tolist(), enumerate(gt_labels.tolist())):
#     gt_bb = torch.squeeze(gt_bb)
#     gt_dict[str(label_gt)].append(gt_bb[idx2].numpy())
#     if score > 0.7:
#         pred_dict[str(label_pred)].append(pred_bb[idx1].numpy())

# print(len(gt_dict.values()))

def relative_faulty_iou(gt_labels: torch.Tensor, 
                        _gt_bbs: torch.Tensor, 
                        pred_labels: torch.Tensor, 
                        pred_bb: torch.Tensor, 
                        pred_scores: torch.Tensor, 
                        correspondance_dict: Dict[str, np.ndarray]):
    pass

def relative_iou(gt_labels: torch.Tensor, _gt_bbs: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor, pred_scores: torch.Tensor):
    score_per_label = list()


    pred_dict, gt_dict = setup_dicts(pred_labels, pred_scores, pred_bb, gt_labels, _gt_bbs)

    correspondence = dict()

    for label in list(pred_dict.keys()):

        indices_per_label = list()
        # bb_id = 0
        if label in list(gt_dict.keys()):
            pred_bbs = np.array(pred_dict[label])
            # print(f'pred_bbs: {pred_bbs}')
            gt_bbs = gt_dict[label]
            # print(f'gt_bbs: {gt_bbs}')
            # print(f'label: {label}')
            for gt_bb in gt_bbs:
                # compute the array-wise subtraction between the current gt_bb and each pred_bb corresponding to the same label
                #distances = np.abs(gt_bb - pred_bbs)
                disatnces1 = np.linalg.norm(gt_bb[0:2] - pred_bbs[:,0:2], axis=1)
                disatnces2 = np.linalg.norm(gt_bb[2:4] - pred_bbs[:,2:4], axis=1)
                
                # sum all distances
                buffer = disatnces1 + disatnces2
                # buffer = np.sum(distances, axis = 1)
                
                
                candidate_idx = np.argmin(buffer)

                # take the array correspinding to the lowest distance from the reference gt_bb 
                candidate_bb = pred_bbs[candidate_idx]

                # compute the score between the nearest bb and the gt_bb
                score = compute_iou(gt_bb, candidate_bb)

                # save result
                score_per_label.append((label, score))

                lab_bb_id = str(label) +'_'+ str(candidate_idx)

                correspondence[lab_bb_id] = candidate_bb

                # bb_id += 1
                # delete the already extracted array
                pred_bbs = np.delete(pred_bbs, np.argmin(buffer), axis = 0)

                # pred_bbs[candidate_idx] = np.array([np.nan, np.nan, np.nan, np.nan])
                if len(pred_bbs) == 0:
                    break
    return score_per_label, correspondence

def setup_dicts(pred_labels, pred_scores, pred_bb, gt_labels, _gt_bbs):

    pred_dict = defaultdict(lambda:[])
    gt_dict = defaultdict(lambda:[])
    gt_labels = torch.squeeze(gt_labels)

    for idx, label in enumerate(gt_labels):
        gt_bb = torch.squeeze(_gt_bbs)
        gt_dict[int(label)].append(gt_bb[idx].numpy()) 

    for (idx1, label), score in zip(enumerate(pred_labels.tolist()), pred_scores.tolist()):
        if score > 0.6:
            pred_dict[int(label)].append(pred_bb[idx1].numpy())
            
    return pred_dict, gt_dict

def compute_iou(gt_bb: List[Union[float, float, float, float]], 
                pred_bb: List[Union[float, float, float, float]]):
    # get coordinates
    gt_x1, gt_y1, gt_x2, gt_y2 = extract_coordinates(gt_bb)
    pred_x1, pred_y1, pred_x2, pred_y2 = extract_coordinates(pred_bb)
    # print(f'extract_coordinates(gt_bb): {extract_coordinates(gt_bb)}')

    # intersection box design
    bot_left_x = max(gt_x1, pred_x1)
    bot_left_y = max(gt_y1, pred_y1)
    top_right_x = min(gt_x2, pred_x2)
    top_right_y = min(gt_y2, pred_y2)
    # print(f'intersection box: [{bot_left_x},{bot_left_y}, {top_right_x}, {top_right_y}]')

    # intersection area
    intersection = max(0, top_right_x - bot_left_x + 1) * max(0, top_right_y - bot_left_y + 1)
    # print(f'intersection: {intersection}')

    # independent boxes areas
    area_gt = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    area_pred = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

    union = (area_gt + area_pred - intersection)
    # print(f'union: {union}')

    score = (intersection / union)
    return score


def extract_coordinates(bb):
    return math.floor(bb[0]), math.floor(bb[1]), math.ceil(bb[2]), math.ceil(bb[2])

gt_boxes = torch.tensor([[  3.2700, 266.8500, 404.5000, 475.1000],
         [183.3600, 136.5600, 244.1400, 228.9500],
         [455.9800, 192.5000, 464.5500, 228.0900],
         [453.3100, 252.9700, 461.3300, 286.9000],
         [444.7600, 297.6300, 450.0800, 337.3100],
         [505.9500, 191.0200, 518.1000, 227.8000],
         [487.5100, 199.3300, 494.9900, 227.3800],
         [244.8200, 230.4500, 349.5400, 318.1400],
         [347.3500, 212.3700, 429.8600, 355.3700],
         [460.9100, 191.6700, 490.2400, 227.7100],
         [527.0200, 248.5700, 551.4200, 289.0000],
         [519.3900, 193.4300, 523.4700, 227.6200],
         [497.3900,  55.4300, 501.4700,  82.7900],
         [524.3200,  97.3800, 527.1200, 135.1500],
         [493.3600, 155.7200, 525.9500, 162.0800],
         [454.6600, 245.2700, 503.2100, 257.4100],
         [461.7700, 253.6800, 470.0100, 286.9900]])

gt_labels= torch.tensor([65, 64, 84, 84, 84, 84, 84, 62, 64, 84, 84, 84, 84, 84, 84, 84, 84])

pred_labels = torch.tensor([64, 65, 64, 62, 84, 44, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
        84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
        84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
        84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
        84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 86, 84, 84, 84, 84, 84, 84,
        84, 84, 84, 84, 84, 84, 84, 84, 84, 84])

pred_boxes = torch.tensor([[340.1339, 221.6077, 429.7591, 349.5268],
        [  0.0000, 266.0756, 405.9967, 478.7044],
        [192.4043, 135.9150, 237.1078, 230.9668],
        [250.2832, 234.8118, 335.3478, 315.7598],
        [470.9579,  43.3345, 557.5704,  84.4642],
        [ 96.9386, 192.4095, 113.8222, 230.0073],
        [427.8219, 236.4829, 545.3261, 291.3026],
        [521.8726, 244.8543, 549.8798, 289.6701],
        [524.5253, 145.7858, 553.3793, 181.3221],
        [501.4626, 246.6666, 532.9767, 287.3477],
        [525.0877, 100.3864, 531.8149, 133.9600],
        [482.5672, 254.2570, 497.4354, 287.6950],
        [508.9992, 194.3013, 515.7678, 226.6284],
        [481.9319,  49.6129, 490.2105,  82.9808],
        [527.8839, 194.2191, 534.2521, 226.6432],
        [443.2188, 295.5188, 450.5773, 338.0044],
        [491.0541, 159.8008, 528.4573, 168.2979],
        [454.7747, 250.0709, 462.8939, 286.8847],
        [518.5718, 193.7376, 524.5474, 227.0368],
        [464.2935, 251.5765, 472.4515, 287.1229],
        [482.8809, 191.7688, 555.1828, 228.3226],
        [489.1841, 154.1554, 527.1672, 164.3754],
        [546.9576,  95.7857, 554.1238, 133.0102],
        [532.7328, 100.9667, 538.8700, 133.7028],
        [458.2379, 294.8586, 465.2995, 337.8212],
        [436.7722, 146.6342, 442.8328, 181.6911],
        [513.8615,  53.7952, 520.0452,  82.6069],
        [529.6095,  47.5681, 537.2689,  81.3288],
        [476.7347, 253.0828, 488.0055, 286.9973],
        [501.6687, 196.9353, 508.2618, 226.5795],
        [522.3928,  98.5401, 529.2209, 133.1949],
        [493.0246, 198.3622, 501.6972, 226.6747],
        [492.0356, 167.2968, 531.0712, 175.4024],
        [439.0129, 146.9642, 444.6624, 181.7470],
        [455.8805, 294.0823, 477.3851, 339.2516],
        [477.7389, 252.6260, 506.2898, 287.3392],
        [448.4619, 148.3809, 454.3831, 182.1593],
        [540.3695,  46.8640, 547.5994,  80.8386],
        [530.1760, 194.2482, 536.2922, 226.7990],
        [443.8869, 148.0920, 449.9887, 181.7050],
        [437.9863, 192.6423, 445.4916, 225.4013],
        [470.8947, 151.2498, 477.1269, 181.9358],
        [436.0486, 293.7808, 454.4348, 339.0799],
        [488.8043, 151.8084, 529.7417, 171.6516],
        [448.1158, 295.4652, 455.5539, 338.4948],
        [498.8115, 197.4540, 505.9644, 226.6448],
        [486.1655,  50.2823, 492.8195,  82.7814],
        [506.2398, 195.2427, 512.8107, 226.6355],
        [488.7632, 195.3251, 518.8022, 227.5497],
        [539.2642, 192.8807, 545.1415, 226.4175],
        [441.3979, 147.5600, 447.2555, 181.6711],
        [535.6758, 101.5471, 541.5190, 133.9063],
        [432.3430, 185.5119, 559.4467, 230.3568],
        [472.6936, 151.8125, 479.9407, 182.0299],
        [468.3346, 250.8102, 477.5874, 286.9980],
        [445.9911, 148.2817, 451.9432, 181.8018],
        [435.5918, 246.1537, 507.3662, 288.2725],
        [461.2301, 249.7956, 494.9069, 287.6047],
        [477.4119, 152.6805, 487.3290, 181.5786],
        [450.5773, 148.2250, 456.6136, 182.2126],
        [438.2082, 295.1196, 445.7384, 337.9308],
        [490.7022, 253.6762, 508.5450, 287.2234],
        [490.1550, 240.8959, 548.3022, 288.8169],
        [527.3427, 100.3336, 534.1845, 133.9883],
        [485.7262, 249.4569, 517.6013, 287.7535],
        [514.6533, 193.9026, 520.6387, 226.9770],
        [437.1434, 146.2120, 491.7977, 182.9917],
        [550.2445,  45.1164, 556.9363,  81.2960],
        [455.7024, 148.8791, 461.8008, 181.9970],
        [442.9034, 192.5019, 449.3795, 225.6328],
        [432.7118, 191.7039, 438.4740, 225.4808],
        [524.1970, 194.4333, 529.5322, 226.4587],
        [430.9003, 293.4576, 438.0663, 339.4480],
        [545.6133, 192.1142, 552.5759, 226.9353],
        [442.1221, 292.7401, 491.9806, 339.2183],
        [465.1466, 295.5880, 472.4584, 337.4113],
        [489.1578, 198.5337, 497.5614, 226.6069],
        [433.3962, 146.0094, 440.2656, 181.7659],
        [503.7529,  54.3332, 510.8672,  82.7240],
        [490.7527, 164.5123, 533.1631, 180.5799],
        [434.7413, 247.2863, 443.6866, 286.2046],
        [462.1134, 151.6550, 469.5918, 181.6365],
        [538.4536,  46.9956, 544.6812,  81.1845],
        [206.3325, 206.2604, 230.5700, 228.6507],
        [522.0057,  97.9821, 538.5878, 134.3061],
        [494.5493,  53.1913, 501.2938,  82.7416],
        [492.0782, 152.2433, 525.6089, 160.6099],
        [450.7191, 247.8527, 457.7810, 286.0146],
        [468.5530, 149.8428, 490.3524, 182.8965],
        [539.9240,  99.8590, 549.8005, 133.3466],
        [472.5412, 295.3162, 479.2926, 337.5629],
        [460.2801, 251.5608, 467.2172, 287.0623],
        [548.3218,  45.4160, 554.3557,  81.0592],
        [445.9545, 246.5650, 464.5564, 287.2766],
        [432.3411, 190.0525, 449.8808, 226.1590],
        [496.2315, 236.9727, 543.8854, 246.5880],
        [488.7079,  51.3936, 495.4221,  83.1041],
        [448.7055, 192.8940, 456.1789, 225.8328],
        [432.3453, 145.1944, 448.4774, 182.0012],
        [447.6265, 247.9399, 454.5610, 285.6614]])

pred_scores = torch.tensor([0.9941, 0.9935, 0.9892, 0.9867, 0.9551, 0.9381, 0.9206, 0.8751, 0.8639,
        0.8567, 0.8372, 0.6719, 0.6587, 0.6221, 0.6217, 0.4903, 0.4504, 0.4049,
        0.3903, 0.3431, 0.2995, 0.2871, 0.2854, 0.2841, 0.2763, 0.2620, 0.2502,
        0.2321, 0.2216, 0.2145, 0.1935, 0.1814, 0.1796, 0.1781, 0.1780, 0.1663,
        0.1508, 0.1437, 0.1431, 0.1415, 0.1405, 0.1374, 0.1347, 0.1332, 0.1328,
        0.1319, 0.1290, 0.1288, 0.1284, 0.1265, 0.1264, 0.1259, 0.1257, 0.1153,
        0.1124, 0.1091, 0.1067, 0.0996, 0.0982, 0.0954, 0.0954, 0.0947, 0.0930,
        0.0920, 0.0878, 0.0869, 0.0861, 0.0860, 0.0849, 0.0839, 0.0805, 0.0785,
        0.0782, 0.0768, 0.0747, 0.0730, 0.0716, 0.0710, 0.0653, 0.0640, 0.0638,
        0.0626, 0.0619, 0.0579, 0.0568, 0.0563, 0.0551, 0.0549, 0.0540, 0.0522,
        0.0522, 0.0504, 0.0501, 0.0499, 0.0498, 0.0497, 0.0496, 0.0495, 0.0494, 0.0493])


# scores, correspondance_dict = relative_iou(gt_labels=gt_labels, _gt_bbs=gt_boxes, pred_labels=pred_labels, pred_bb=pred_boxes, pred_scores=pred_scores)
# print(scores)
# print(correspondance_dict.keys())
metric = MeanAveragePrecision()
preds = [dict(
    boxes = pred_boxes,
    labels = pred_labels,
    scores = pred_scores
)]
targets = [dict(
    boxes = gt_boxes,
    labels = gt_labels
)]
metric.update(preds, targets)

score = metric.compute()

print(score)