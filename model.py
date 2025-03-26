from diffusers import StableDiffusionPipeline
import torch

def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipeline
