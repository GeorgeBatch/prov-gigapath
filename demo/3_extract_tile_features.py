import os
import torch
import timm

from gigapath.pipeline import run_inference_with_tile_encoder


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


image_paths = [
    os.path.join(specific_slide_tiles_dir, img)
    for img in os.listdir(specific_slide_tiles_dir)
    if img.endswith(".png")
]
print(f"Found {len(image_paths)} image tiles")


tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
tile_encoder_outputs = run_inference_with_tile_encoder(
    image_paths, tile_encoder, batch_size=32
)

for k in tile_encoder_outputs.keys():
    print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")

# save features and coordinates pytorch tensors
torch.save(
    tile_encoder_outputs["tile_embeds"],
    os.path.join(specific_slide_features_dir, "tile_embeds.pt"),
)
torch.save(
    tile_encoder_outputs["coords"],
    os.path.join(specific_slide_features_dir, "coords.pt"),
)
print("Saved features and coordinates to", specific_slide_features_dir)
