import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import save_image
from torchvision import transforms as tfms
from PIL import Image
from torch.utils.data.dataloader import default_collate
from utils import _DMADataset
from utils import parse_args_and_config, norm
from tqdm import tqdm

def load_model(model_id, device):
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype=torch.float16)
    return vae, tokenizer, text_encoder, unet

def load_image(image_path, size, device):
    img = Image.open(image_path).convert('RGB')
    if size:
        img = img.resize((size,size))
    return tfms.functional.to_tensor(img).unsqueeze(0).to(device, dtype=torch.float16) * 2 - 1


def generate_text_embeddings(tokenizer, text_encoder, device, prompt="", negative_prompt=""):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        negative_prompt, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

def generate_latent(vae, noise_scheduler, x, size, batch_size, device):
    x = x.to(device, dtype=torch.float16)  
    latents = vae.encode(x)
    latents = 0.18215 * latents.latent_dist.sample()
    latents = latents * noise_scheduler.init_noise_sigma
    return latents

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0].nelement() > 0]
    if not batch:
        return torch.tensor([]), []
    return default_collate(batch)

@torch.no_grad()
def pipe(args):
    device = args.device
    model_id = args.diffusion_ckpt
    load_size = args.loadSize
    prompt = args.prompt
    negative_prompt = args.n_prompt
    guidance_scale = args.guidance_scale
    batch_size = args.batch_size
    t = args.selected_step
    
    vae, tokenizer, text_encoder, unet = load_model(model_id, device)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    
    text_embeddings = generate_text_embeddings(tokenizer, text_encoder, device, prompt=prompt, negative_prompt=negative_prompt)
    
    dataset = _DMADataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=int(args.num_threads), collate_fn=custom_collate_fn)
    for batch in tqdm(dataloader):
        x, save_paths = batch
        x = x.to(device)
        latents = generate_latent(vae, noise_scheduler, x, load_size, batch_size, device)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        text_embeddings_tmp = text_embeddings.repeat(latent_model_input.shape[0]//2,1,1)

        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_tmp).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        avid_noises = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        avid_noises = 1 / 0.18215 * avid_noises
        images = vae.decode(avid_noises).sample
        
            
        for idx, item in enumerate(images):
            save_image(item, save_paths[idx], normalize=True)

    
if __name__ == '__main__':
    print('***********************')
    args, _ = parse_args_and_config()
    pipe(args)
