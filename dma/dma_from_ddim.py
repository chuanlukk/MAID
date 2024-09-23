"""
Modified from DNF/compute_dnf.py
"""
import os
import numpy as np

import torch

from tqdm import tqdm


from torchvision.utils import save_image

from diffusion import Model
from utils import _DMADataset
from utils import parse_args_and_config, norm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0].nelement() > 0]
    if not batch:
        return torch.tensor([]), []
    return default_collate(batch)


if __name__ == '__main__':

    args, config = parse_args_and_config()

    diffusion = Model(config)
    diffusion.load_state_dict(torch.load(args.diffusion_ckpt))
    diffusion = diffusion.to(args.device)
    diffusion.eval()

    dataset = _DMADataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=int(args.num_threads), collate_fn=custom_collate_fn)

    for batch in tqdm(dataloader):
        x, save_paths = batch
        x = x.to(args.device)
        try:
            # inverse diffusion
            with torch.no_grad():
                n = x.size(0) # batch
                t = torch.zeros(n).to(x.device) # timestep
                dma = diffusion(x,t)
                
            for idx, item in enumerate(dma):
                save_image(item, save_paths[idx], normalize=False)
        except Exception as e:
            print(f"Error processing image: {e}")
