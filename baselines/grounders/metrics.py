import numpy as np
import torch
import cv2


# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [
#         torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output


def segmentation_metrics(preds, masks, device):
    iou_list = []
    for pred, mask in zip(preds, masks):
        # pred: (H, W): bool, mask: (H, W): bool
        # iou
        inter = np.logical_and(pred, mask)
        union = np.logical_or(pred, mask)
        iou = np.sum(inter) / (np.sum(union) + 1e-6)
        iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: IoU={:.2f}'.format(100. * iou.item())
    return head + temp, {'iou': iou.item(), **prec}