from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import torch

def fine_tune_model():
    dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    def collate_fn(examples):
        texts = [example["text"] for example in examples]
        images = [example["image"] for example in examples]
        return {"texts": texts, "images": images}

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=1e-4)

    for epoch in range(3):
        for batch in train_dataloader:
            texts, images = batch["texts"], batch["images"]
            loss = pipeline(texts, images).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    pipeline.save_pretrained("./fine_tuned_model")
    print("Model fine-tuned and saved.")

if __name__ == "__main__":
    fine_tune_model()
