import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from tqdm import tqdm 

import clip as clip_module

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import warnings
warnings.filterwarnings('ignore')

from .prompts import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
	

class SamAndClipGrounder(nn.Module):

	def __init__(self, cfg, device):
		super().__init__()
		self.device = device
		self.cfg = cfg
		
		# load Segment-Anything Model]
		sam = sam_model_registry["default"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth").to(device)
		sam.eval()
		self.mask_generator = SamAutomaticMaskGenerator(
			model=sam,
			points_per_side=self.cfg["sam"]["points_per_side"],
			pred_iou_thresh=self.cfg["sam"]["pred_iou_thresh"],
			stability_score_thresh=self.cfg["sam"]["stability_score_thresh"],
			crop_n_layers=self.cfg["sam"]["crop_n_layers"],
			crop_n_points_downscale_factor=self.cfg["sam"]["crop_n_points_downscale_factor"],
			min_mask_region_area=self.cfg["sam"]["min_mask_region_area"],  # Requires open-cv to run post-processing
		)

		# load CLIP
		self.clip, self.preprocess = clip_module.load(self.cfg["clip"]["visual"], device=device, jit=False)
		self.clip.eval()
		self.prompt_engineering = self.cfg["clip"]["prompt_engineering"]
		self.this_is = self.cfg["clip"]["this_is"]

	def build_text_embedding(self, categories, prompt_engineering=True, this_is=True):
		if prompt_engineering:
			templates = multiple_templates
		else:
			templates = single_template

		with torch.no_grad():
			all_text_embeddings = []
			# print('Building text embeddings...')
			for category in categories:
				texts = [
					template.format(processed_name(category['name'], rm_dot=True),
						article=article(category['name']))
				for template in templates]
				if this_is:
					texts = [
							'This is ' + text if text.startswith('a') or text.startswith('the') else text 
							for text in texts
							]
				texts = clip_module.tokenize(texts).to(self.device) #tokenize
				text_embeddings = self.clip.encode_text(texts) #embed with text encoder
				text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
				text_embedding = text_embeddings.mean(dim=0) #average accross prompt templates
				text_embedding /= text_embedding.norm()
				all_text_embeddings.append(text_embedding)
			all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
	    
		return all_text_embeddings.to(self.device).T

	@torch.no_grad()
	def predict(self, img, queries, 
				with_mask = False
		):
		#img: PIL Image or np.array, queries: ;-separated text to ground
		if not isinstance(img, np.ndarray): img = np.asarray(img)
		if with_mask: img_blur = cv2.GaussianBlur(img, (101, 101), 100)

		with torch.no_grad():
			masks = self.mask_generator.generate(img)
		if self.device == "cuda":
			torch.cuda().empty_cache()

		object_image_features = []
		for m in masks:

			bbox = m['bbox']
			x, y, w, h = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
			crop_box = img[y:y+h, x:x+w, :]
			image_input = self.preprocess(Image.fromarray(crop_box)).to(self.device)

			# ReCLIP style - both mask and crop
			if with_mask:
				segm = m['segmentation']
				mask = np.zeros_like(img)
				# mask[segm==True] = img[segm==True]
				mask = np.where(segm==True, img, img_blur)
				mask_input = self.preprocess(Image.fromarray(mask)).to(self.device)

				image_input = torch.stack([mask_input, image_input])
			else:
				image_input = image_input.unsqueeze(0)

			with torch.no_grad():
				image_features = self.clip.encode_image(image_input)

			image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
			object_image_features.append(image_features_norm.mean(0)) # (2, C) -> (C,)

		# average accross mask-crop features
		object_image_features = torch.stack(object_image_features)	# (N, C))

		# expected ;-seperated
		query_names = [x.strip() for x in queries.split(';')]
		queries = [{'name': item, 'id': idx+1,} for idx, item in enumerate(query_names)]

		# (Q, C)
		text_embedding = self.build_text_embedding(queries, self.prompt_engineering, self.this_is)

		similarities = object_image_features @ text_embedding.T # (N, Q)
		most_similar_indices =  similarities.argmax(0)	# (Q,)

		return_masks = [masks[i]['segmentation'] for i in most_similar_indices]
		
		return return_masks, query_names

	def forward(self):
		raise NotImplementedError


def make_model(cfg=None, device=DEVICE):
	if cfg is None:
		# default config
		cfg = {
			'sam': {'points_per_side': 64,
					'pred_iou_thresh': .9,
					'stability_score_thresh': .98,
					'crop_n_layers':1,
					'crop_n_points_downscale_factor':2,
					'min_mask_region_area':10000},
			'clip': {'visual': 'RN50',
					 'prompt_engineering': False,
					 'this_is': True}
		}
	else:
		cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg

	return SamAndClipGrounder(cfg, device)