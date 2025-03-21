from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# Preprocess NFT image → Canny Edges
def apply_canny_edge(nft_path):
    nft = Image.open(nft_path).convert("RGB").resize((512, 512))
    nft_np = np.array(nft)
    nft_gray = cv2.cvtColor(nft_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(nft_gray, 100, 200)
    edge_img = Image.fromarray(edges).convert("RGB")
    return edge_img

# -------------------------------
# Load ControlNet + Base Model + Inject LoRA
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load trained LoRA weights
pipe.unet.load_attn_processors("./lora_output")

# -------------------------------
# Inference
nft_edge_image = apply_canny_edge("./path/to/NFT_image.png")

prompt = "Hoodie with cyberpunk NFT art, centered chest print, olive background, minimalist branding."

output = pipe(
    prompt=prompt,
    image=nft_edge_image,
    num_inference_steps=40
).images[0]

output.save("final_nft_hoodie_output.png")
print("✅ Generated NFT Hoodie Design!")