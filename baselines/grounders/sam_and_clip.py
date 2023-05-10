import numpy as np
import torch
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from tqdm import tqdm 

import clip

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import warnings
warnings.filterwarnings('ignore')

from .prompts import build_text_embedding

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
	

class SamAndClipGrounder:

	def __init__(self, cfg, device):
		self.device = device
		self.cfg = cfg
		
		# load Segment-Anything Model
		sam = sam_model_registry["default"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
		sam = sam.eval().to(device)
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
		self.clip, self.preprocess = clip.load(self.cfg["clip"]["visual"], device=device, jit=False)
		self.clip = clip.eval().to(device)
		self.prompt_engineering = self.cfg["clip"]["prompt_engineering"]
		self.this_is = self.cfg["clip"]["this_is"]

	@torch.no_grad()
	def predict(img, queries):
		#img: PIL Image or np.array, queries: ;-separated text to ground
		if isinstance(img, Image): img = np.asarray(img)

		masks = self.mask_generator.generate(img)
		if self.device == torch.device("cuda"):
			torch.cuda().empty_cache()

		object_image_features = []
		for m in masks:
			# ReCLIP style - both mask and crop
			segm = m['segmentation']
			mask = np.zeros_like(img)
			mask[segm==True] = img[segm==True]
			mask_input = preprocess(Image.fromarray(mask)).to(self.device)

			bbox = m['bbox']
			x, y, w, h = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
			crop_box = image[y:y+h, x:x+w, :]
			box_input = preprocess(Image.fromarray(crop_box)).to(self.device)

			image_inputs = torch.stack([mask_input, box_input])
			image_features = self.clip.encode_image(image_inputs)
			image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
			object_image_features.append(image_features_norm)
		object_image_features = torch.stack(object_image_features)

		# expected ;-seperated
		query_names = [x.strip() for x in queries.split(';')]
		queries = [{'name': item, 'id': idx+1,} for idx, item in enumerate(query_names)]

		text_embedding = build_text_embedding(queries, self.prompt_engineering, self.this_is)

		similarities = object_image_features @ text_embedding.T
		most_similar_indices =  similarities.mean(1).argmax(0)

		result = []
		for q, i in zip(query_names, most_similar_indices):
			m = masks[i]['segmentation']
			result.append(q, m)
		
		return result


def default_model(device=DEVICE):
	cfg = {
		'sam': {'points_per_side': 64,
				'pred_iou_thresh': .9,
				'stability_score_thresh': .98,
				'crop_n_layers':1,
				'crop_n_points_downscale_factor':2,
				'min_mask_region_area':10000}
		'clip': {'visual': 'RN50',
				 'prompt_engineering': True,
				 'this_is': True}
	}
	return SamAndClipGrounder(cfg, device)