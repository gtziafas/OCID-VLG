import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from skimage.filters import gaussian

from baselines.graspers.grconvnet.network.inference.post_process import post_process_output
from baselines.graspers.grconvnet.network.inference.models import get_network
from baselines.graspers.grconvnet.network.utils.visualisation.plot import plot_results
from baselines.graspers.grconvnet.network.utils.data.camera_data import CameraData
from baselines.graspers.grconvnet.network.utils.dataset_processing.grasp import detect_grasps, GraspRectangles

# from baselines.graspers.grconvnet.network.hardware.device import get_device
# from baselines.graspers.grconvnet.network.inference.post_process import post_process_output
# from baselines.graspers.grconvnet.network.utils.visualisation.plot import plot_results
# from baselines.graspers.grconvnet.network.inference.post_process import post_process_output
# # from baselines.graspers.grconvnet.network.inference.grasp_generator import GraspGenerator

GR_CONVNET_PATH = 'baselines/graspers/grconvnet/trained_models/GR_ConvNet'
#DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


# trans = torch.nn.Sequential(
#     transforms.Resize(224)
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# )


def expand_masks(masks, exp_factor = 50):
    result  = []
    for m in masks:
        cs, _ = cv2.findContours(255 * m.astype(np.uint8), 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #assert len(cs) == 1
        if len(cs) == 0:
            result.append(np.zeros_like(m))
            continue
        cs = sorted(cs, key=lambda c: cv2.contourArea(c), reverse=False)[0]
        x,y,w,h = cv2.boundingRect(cs[0])
        x1 = max(x - exp_factor, 0)
        y1 = max(y - exp_factor, 0)
        x2 = min(x + w + exp_factor, m.shape[1])
        y2 = min(y + h + exp_factor, m.shape[0])
        m_exp = np.zeros_like(m)
        m_exp[y1:y2,x1:x2] = True
        result.append(m_exp)
    return torch.from_numpy(np.stack(result))  


# from grasp rect to grasp points format
def rect_to_pts(grasp_rectangles):
    boxes = []
    for rect in grasp_rectangles:
        center_x, center_y, width, height, theta, _ = rect
        box = ((center_x, center_y), (width, height), -(theta+180))
        box = cv2.boxPoints(box)
        box = np.intp(box)
        boxes.append(box)
    return boxes


class Grasper(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        _net = get_network(self.cfg['network_name'])
        self.network_name = self.cfg['network_name']
        self.net_path = self.cfg['net_path']
        self.device = self.cfg['device']
        self.model = _net(input_channels=4, dropout=False, prob=0., channel_size=32).to(self.device)
        self._load_net()
        self.model.eval()
        self.width, self.height = self.cfg['width'], self.cfg['height']
        self.cam_data = CameraData(width=self.cfg['width'], height=self.cfg['height'], 
            include_depth=True, include_rgb=True)
        self.n_grasps = self.cfg['n_grasps']
        self.projection = self.cfg['projection']
        self.resize = self.cfg['resize']
        self.expand_factor = self.cfg['expand_factor']
        self.resize_tf = transforms.Compose([transforms.Resize([self.resize, self.resize]) if self.resize else torch.nn.Identity()])
        self.cam_data = CameraData(output_size=self.resize)

        # convert to cv2-box like
        self.cvt_grasps = lambda gs: [list(map(int, [g.center[1],g.center[0],g.length,g.width,
            np.rad2deg(g.angle), 0])) for g in gs]

    def _load_net(self):
        cwd = os.getcwd()
        if self.network_name.startswith('grconvnet'):
            os.chdir(GR_CONVNET_PATH)
            print(f'Loading {self.network_name} from {self.net_path}')
            self.model.load_state_dict(torch.load(self.net_path,
                map_location=self.device).state_dict())
            os.chdir(cwd)
        else:
            raise ValueError("The selected network has not been implemented yet -- please choose another network!")
            
    def forward(self):
        raise NotImplementedError

    # def preprocess_rgbd(self, rgb, depth):
    #     #depth = 1 - depth / depth.max(axis=(1,2), keepdims=True)
    #     depth = 1. - depth
    #     for x_rgb, x_d in zip(rgb, depth):
    #         x, depth_img, rgb_img = self.cam_data.get_data(x_rgb, x_d)

    #     return x

    def preprocess_rgbd(self, rgb, depth):
        depth = 1 - depth / depth.max(axis=(1,2), keepdims=True)
        depth = np.expand_dims(depth, axis=-1)
        rgb = rgb / 255.
        rgb -= rgb.mean(axis=(1,2,3), keepdims=True)
        rgbd = np.concatenate([rgb, depth], axis=-1)
        rgbd = torch.from_numpy(rgbd).permute(0,3,1,2).float()
        return rgbd

    def project_segm_in_rgbd(self, rgb, depth, mask):
        # (B,H,W,C), mask: bool
        #mask_3 = mask.unsqueeze(-1).repeat(1,1,3)
        mask_3 = np.tile(np.expand_dims(mask, axis=-1), (1,1,3))
        #rgb_masked = np.where(mask_3==True, rgb, 0xff * np.ones_like(rgb))
        #depth_masked = np.where(mask==True, depth, depth.max())
        rgb_masked = np.where(mask_3 > 0.5, rgb, 0xff * np.ones_like(rgb))
        depth_masked = np.where(mask > 0.5, depth, depth.max())
        return rgb_masked, depth_masked

    def resize_batch(self, rgb, depth, mask):
        rgb = np.stack([np.asarray(self.resize_tf(Image.fromarray(x))) for x in rgb])
        depth = np.stack([np.asarray(self.resize_tf(Image.fromarray(x))) for x in depth])
        mask = np.stack([np.asarray(self.resize_tf(Image.fromarray(x))) for x in mask])

    def predict(self, rgb, depth, mask, n_grasps=None):
        if len(rgb.shape) == 3:
            # unsqueeze for batched inference
            #mask = gaussian(mask, 3.0, preserve_range=True)
            # kernel = np.ones((13,13),np.uint8)
            # mask = cv2.erode(255  * mask.astype(np.uint8),
            #     kernel, iterations = 1).astype(np.float32) / 255.
            mask = mask.astype(np.float32)
            rgb = np.expand_dims(rgb, axis=0)
            depth = np.expand_dims(depth, axis=0)
            mask = np.expand_dims(mask, axis=0)

        n_grasps = n_grasps if n_grasps is not None else self.n_grasps

        if self.projection in ["before", "combined"]:
            # keep only mask of target object
            rgb, depth = self.project_segm_in_rgbd(rgb, depth, mask)

        # elif self.projection == "combined":
        #     # keep extended bounding box to include neighbor info
        #     rgb, depth = self.project_segm_in_rgbd(rgb, depth, 
        #             expand_masks(mask, self.expand_factor))

        if self.network_name.startswith('grconvnet'):
            
            xc = self.preprocess_rgbd(rgb, depth)
            
            with torch.no_grad():
                xc = xc.to(self.device)
                pred = self.model.predict(xc)
        else:
            raise ValueError("The selected network has not been implemented yet -- please choose another network!")

        if self.projection in ["after", "combined"]:
            mask_pt = torch.from_numpy(mask).float()
            # pred['pos'] = torch.from_numpy(mask).float().unsqueeze(1)
            #pred['pos'] = mask_pt * pred['pos'] 
            pred['pos'] =torch.where(mask_pt == 1, pred['pos'], 0.)
            pred['cos'] = torch.where(mask_pt == 1, pred['cos'], 0.)
            pred['sin'] = torch.where(mask_pt == 1, pred['sin'], 0.)
            pred['width'] = torch.where(mask_pt == 1, pred['width'], 0.)

        grasps = []
        out_batch = {'q': [], 'ang': [], 'w': []}
        for pos, cos, sin, w, m in zip(pred['pos'], pred['cos'], pred['sin'], pred['width'], mask):

            q_img, ang_img, width_img = post_process_output(pos, cos, sin, w)

            # if self.projection in ["after", "combined"]:
            #     #mask = torch.from_numpy(mask)
            #     q_img = np.where(m==True, q_img, 0.)
            #     ang_img = np.where(m==True, ang_img, 0.)
            #     width_img = np.where(m==True, width_img, 0.)

            grasps.append(detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps))
            out_batch['q'].append(q_img)
            out_batch['ang'].append(ang_img)
            out_batch['w'].append(width_img)
            
        return grasps, {k: np.stack(im).squeeze() for k, im in out_batch.items()}

    def calculate_iou_match(self, pred_grasps, gt_grasp_rects, 
                threshold=0.25, keep_top=1):

        def _iou(g_pred, gt_box):
            canvas = np.zeros((480,640), dtype=np.uint8)
            g_pred_msk = cv2.drawContours(canvas.copy(), [g_pred], 0, 0xff, -1)
            gt_msk = cv2.drawContours(canvas.copy(), [gt_box.astype(int)], 0, 0xff, -1)
            union = np.bitwise_or(g_pred_msk, gt_msk).sum()
            intersection = np.bitwise_and(g_pred_msk, gt_msk).sum()
            return intersection / union

        pred_grasps = sorted(pred_grasps, key=lambda g: g.quality, reverse=True)
        pred_grasps = pred_grasps[:min(len(pred_grasps), keep_top)]

        pred_grasps = self.cvt_grasps(pred_grasps)
        pred_pts = rect_to_pts(pred_grasps)
        gt_pts = rect_to_pts(gt_grasp_rects)

        IoUs = []
        for g, gpts in zip(pred_grasps, pred_pts):
            g_angle = g[-2]
            max_iou = 0
            for gt_rect, gt_box in zip(gt_grasp_rects, gt_pts):
                gt_angle = gt_rect[-2]
                if abs(g_angle - gt_angle) > 30:
                    continue
                max_iou = max(max_iou, _iou(gpts, gt_box))
            IoUs.append(max_iou > threshold)

        return bool(sum(IoUs))


def make_grasper(cfg=None, device=DEVICE):
    if cfg is None:
        # default config
        cfg = {
            'net_path' : 'jacquard-rgbd-grconvnet3-drop0-ch32/epoch_42_iou_0.93',
            'width' : 640, 'height': 480,
            'device' : device,
            'n_grasps' : 1,
            'network_name' : 'grconvnet3',
            'projection': "before",
            'resize': 224,
            'expand_factor': 50
        }
    else:
        cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
    return Grasper(cfg)