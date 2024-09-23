DATAROOT="--dataroot path/to/your/dataset"
BS="--batch_size 100"

DIFF="--diffusion_ckpt path/to/your/model-2388000_bedroom.ckpt"
# DIFF="--diffusion_ckpt path/to/your/CompVis--stable-diffusion-v1-4"

POSTFIX="--postfix _ddim"
python dma_from_ddim.py $DATAROOT $POSTFIX $BS $DIFF

# POSTFIX="--postfix _sd"
# python dma_from_sd.py $DATAROOT $POSTFIX $BS $DIFF
