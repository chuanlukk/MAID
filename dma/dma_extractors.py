import torch
from torchvision.utils import save_image
from tqdm import tqdm
import traceback
import sys

from utils import _DMADataset, custom_collate_fn

# Import the libraries required by each model
from diffusion import Model as DDIMModel
from diffusers import IFImg2ImgPipeline, DiffusionPipeline, StableDiffusionPipeline, DiTPipeline, UNet2DModel, VQModel

class DMAExtractor:
    def __init__(self, args, config=None):
        self.args = args
        if config:
            self.config = config
        self.device = args.device
        self.dataset = _DMADataset(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=args.batch_size, 
            num_workers=int(args.num_threads), 
            collate_fn=custom_collate_fn
        )
    
    def load_model(self):
        raise NotImplementedError("load_model() must be implemented in subclass")
    
    @torch.no_grad()
    def extract_dma(self):
        raise NotImplementedError("extract_dma() must be implemented in subclass")
    
    def handle_exception(self, e):
        print(f"Error processing a batch: {e}")
        traceback.print_exc(file=sys.stdout)

class DDIMExtractor(DMAExtractor):
    def load_model(self):
        self.diffusion = DDIMModel(self.config)
        self.diffusion.load_state_dict(torch.load(self.args.diffusion_path, map_location=self.device, weights_only=True))
        self.diffusion = self.diffusion.to(self.device)
        self.diffusion.eval()
    
    def extract_dma(self):
        for batch in tqdm(self.dataloader, desc="Extracting DMA with DDIM"):
            x, save_paths = batch
            if x.nelement() == 0: continue
            x = x.to(self.device)
            B = x.size(0)
            t = torch.zeros(B, device=self.device)
            # inverse diffusion
            noise_pred = self.diffusion(x, t)  
            for dma_tensor, out_path in zip(noise_pred, save_paths):
                save_image(dma_tensor, out_path)

class IFExtractor(DMAExtractor):
    def load_model(self):
        self.pipe = IFImg2ImgPipeline.from_pretrained(
            self.args.diffusion_path,
            torch_dtype=torch.float16
        ).to(self.device)
        # Encode prompt
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=self.args.prompt, 
            do_classifier_free_guidance=True if self.args.guidance_scale > 1 else False,
            device=self.device,
        )
    
    def extract_dma(self):
        for batch in tqdm(self.dataloader, desc="Extracting DMA with IF"):
            x, save_paths = batch
            if x.nelement() == 0: continue
            B = x.size(0)
            x = x.to(self.device, dtype=torch.float16)
            t = torch.zeros(B, device=self.device, dtype=torch.long)
            current_prompt_embeds = self.prompt_embeds.repeat_interleave(B, dim=0)
            noise_pred = self.pipe.unet(
                x,
                t,
                current_prompt_embeds,
                return_dict=False
            )[0]
            dma = noise_pred.chunk(2, dim=1)[0]  # noise_pred and predicted_variance
            for dma_tensor, out_path in zip(dma, save_paths):
                save_image(dma_tensor, out_path)

class LDMExtractor(DMAExtractor):
    def load_model(self):
        self.unet = UNet2DModel.from_pretrained(
            self.args.diffusion_path, 
            subfolder="unet",
            torch_dtype=torch.float16
            ).to(self.device)
        self.vqvae = VQModel.from_pretrained(
            self.args.diffusion_path, 
            subfolder="vqvae",
            torch_dtype=torch.float16
            ).to(self.device)
    
    def extract_dma(self):
        for batch in tqdm(self.dataloader, desc="Extracting DMA with LDM"):
            x, save_paths = batch
            if x.nelement() == 0: continue
            B = x.size(0)
            x = x.to(self.device, dtype=torch.float16)
            t = torch.zeros(B, device=self.device, dtype=torch.long)
            # Encode images to latent space
            img_latents = self.vqvae.encode(x).latents
            noise_pred = self.unet(img_latents, t)["sample"]
            # Decode latent to pixel space
            dma = self.vqvae.decode(noise_pred).sample
            for dma_tensor, out_path in zip(dma, save_paths):
                save_image(dma_tensor, out_path)

class SDExtractor(DMAExtractor):
    def load_model(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.args.diffusion_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.scaling_factor = self.pipe.vae.config.scaling_factor
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=self.args.prompt,                     
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True if self.args.guidance_scale > 1 else False,
        )
    
    def extract_dma(self):
        for batch in tqdm(self.dataloader, desc="Extracting DMA with SD"):
            x, save_paths = batch
            if x.nelement() == 0: continue
            x = x.to(self.device, dtype=torch.float16)
            B = x.size(0)
            t = torch.zeros(B, device=self.device, dtype=torch.long)
            latents = self.pipe.vae.encode(x).latent_dist.sample() * self.scaling_factor
            noise_pred = self.pipe.unet(latents, t, encoder_hidden_states=self.prompt_embeds.repeat(B, 1, 1)).sample
            dma = self.pipe.vae.decode(noise_pred / self.scaling_factor).sample
            for dma_tensor, out_path in zip(dma, save_paths):
                save_image(dma_tensor, out_path, normalize=True)

class DITExtractor(DMAExtractor):
    def load_model(self):
        self.pipe = DiTPipeline.from_pretrained(
            self.args.diffusion_path,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.scaling_factor = self.pipe.vae.config.scaling_factor
        self.latent_channels = self.pipe.transformer.config.in_channels
    
    def extract_dma(self):
        for batch in tqdm(self.dataloader, desc="Extracting DMA with DiT"):
            x, save_paths = batch
            if x.nelement() == 0:
                continue
            B = x.size(0)
            x = x.to(self.device, dtype=self.pipe.vae.dtype)
            t = torch.zeros(B, device=self.device)
            class_labels = torch.tensor([1000] * B, device=self.device)
            latents = self.pipe.vae.encode(x).latent_dist.sample() * self.scaling_factor
            noise_pred = self.pipe.transformer(
                latents, timestep=t, class_labels=class_labels
            ).sample
            if self.pipe.transformer.config.out_channels // 2 == self.latent_channels:
                model_output, _ = torch.split(noise_pred, self.latent_channels, dim=1)
            else:
                model_output = noise_pred
            dma = self.pipe.vae.decode(model_output / self.scaling_factor).sample
            for dma_tensor, out_path in zip(dma, save_paths):
                save_image(dma_tensor, out_path, normalize=True)