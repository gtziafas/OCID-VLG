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
	def predict_grasp(self, imgs, masks):
		# imgs: (B, C, H, W), # masks: (B, H, W)
		

	@torch.no_grad()
	def predict(self, ):
		pass

	

	