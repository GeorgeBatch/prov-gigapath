import os
import torch

import gigapath.slide_encoder as slide_encoder
from gigapath.pipeline import run_inference_with_slide_encoder


PROJECT_DIR = "."
local_dir_name = "sample_data"
local_dir = os.path.join(PROJECT_DIR, local_dir_name)

slide_file_name = "PROV-000-000001.ndpi"
slide_hf_path = os.path.join(local_dir_name, slide_file_name)
slide_path = os.path.join(local_dir, "PROV-000-000001.ndpi")


tile_save_dir = os.path.join(local_dir, "outputs/preprocessing")
specific_slide_tiles_dir = f"{tile_save_dir}/output/{slide_file_name}"
os.makedirs(specific_slide_tiles_dir, exist_ok=True)

features_save_dir = os.path.join(local_dir, "outputs/features")
specific_slide_features_dir = f"{features_save_dir}/output/{slide_file_name}"
os.makedirs(specific_slide_features_dir, exist_ok=True)


tile_encoder_outputs = {}
tile_encoder_outputs["tile_embeds"] = torch.load(
    os.path.join(specific_slide_features_dir, "tile_embeds.pt")
)
tile_encoder_outputs["coords"] = torch.load(
    os.path.join(specific_slide_features_dir, "coords.pt")
)


slide_encoder_model = slide_encoder.create_model(
    "hf_hub:prov-gigapath/prov-gigapath",
    "gigapath_slide_enc12l768d",
    1536,
    global_pool=True,  # like in the demo cell above
)

slide_embeds = run_inference_with_slide_encoder(
    slide_encoder_model=slide_encoder_model, **tile_encoder_outputs
)


# save slide features
print(slide_embeds.keys())
for key in slide_embeds.keys():
    torch.save(slide_embeds[key], os.path.join(specific_slide_features_dir, f"slide_embeds_{key}.pt"))
    print(f"Saved slide features to {specific_slide_features_dir}/slide_embeds_{key}.pt")
