import sys
import os

import torch

RENDER_PATH = '/mnt/hdd/render'

from nerfstudio.pipelines.base_pipeline import Pipeline
camera = Pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()

if not os.path.exists(RENDER_PATH):
    os.mkdir(RENDER_PATH)

rays = camera.generate_rays(camera_indices=0)
with torch.no_grad():
    outputs = Pipeline.model.get_outputs_for_camera_ray_bundle(rays)

for rendered_output_name in ["rgb"]:
    output_image = outputs[rendered_output_name].cpu().numpy()
