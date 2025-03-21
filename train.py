import os
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from diffusers.models.attention_processor import LoraAttnProcessor2_0

# ---------------------------
# 1️⃣ Configuration
dataset_path = "./mockup_dataset"
images_dir = os.path.join(dataset_path, "images")
json_path = os.path.join(dataset_path, "dataset.json")
output_dir = "./lora_output"
os.makedirs(output_dir, exist_ok=True)

epochs = 5
batch_size = 2
learning_rate = 1e-4

# ---------------------------
# 2️⃣ Dataset Loader
class NFTMockupDataset(Dataset):
    def __init__(self, json_file, img_dir, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        image = Image.open(img_path).convert("RGB").resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        tokens = self.tokenizer(
            item['prompt'],
            padding='max_length',
            max_length=77,
            return_tensors='pt'
        )

        return {
            'pixel_values': image,
            'input_ids': tokens.input_ids.squeeze()
        }

# ---------------------------
# 3️⃣ Load Tokenizer & Model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# ---------------------------
# 4️⃣ Inject LoRA Layers (NOW CORRECT!)
pipe.unet.set_attn_processor(
    LoraAttnProcessor2_0(
        rank=4,
        scale=1.0
    )
)

# ---------------------------
# 5️⃣ Data Preparation
dataset = NFTMockupDataset(json_path, images_dir, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# 6️⃣ Optimizer
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=learning_rate)

# ---------------------------
# 7️⃣ Training Loop
pipe.unet.train()

for epoch in range(epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        images = batch['pixel_values'].to("cuda", dtype=torch.float16)
        prompt_embeds = pipe.text_encoder(batch['input_ids'].to("cuda"))[0]

        noise = torch.randn_like(images)
        latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
        noisy_latents = latents + noise

        noise_pred = pipe.unet(noisy_latents, torch.tensor([1.0]*images.shape[0], device="cuda"), encoder_hidden_states=prompt_embeds).sample

        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=loss.item())

# ---------------------------
# 8️⃣ Save LoRA Weights
pipe.unet.save_attn_processors(output_dir)
print(f"✅ LoRA weights saved at {output_dir}")