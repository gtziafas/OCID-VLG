import json
import torch
import torch.nn as nn

from .grounders import make_model

class TwoStageBaseline(nn.Module):

	def __init__(self, cfg):
		super().__init__()
		self.cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
		self.grounder = make_grounder(cfg['grounder'])
		self.grasper = make_grasper(cfg['grasper'])

	@torch.no_grad()
	def predict_mask(self, img, queries):
		# img: (3, H, W), # sent: (M,)
		return self.grounder(img, queries) # (M, H, W), bool

	@torch.no_grad()
	def predict_grasps(self, rgb, depth):
		# rgb: (B, 3, H, W), # rgb: (B, H, W)
		pass

	def isolate_mask_in_grasp(self, grasp_out, masks):
		# grasp_out: [pos,qua,ang,wid : (B, H, W)], # masks: (B, H, W)
		grasp_qua_mask = grasp_out["qua"]
		masks = masks.astype(n.float32)		

	@torch.no_grad()
	def predict(self, ):
		pass

	

	