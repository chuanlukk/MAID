# extract_dma.py
import sys
import traceback
from utils import parse_args_and_config
from dma_extractors import DDIMExtractor, IFExtractor, LDMExtractor, SDExtractor, DITExtractor

def get_extractor(diffusion_name, args, config):
    if diffusion_name == 'ddim':
        return DDIMExtractor(args, config)
    elif diffusion_name == 'if':
        return IFExtractor(args)
    elif diffusion_name == 'ldm':
        return LDMExtractor(args)
    elif diffusion_name == 'sd':
        return SDExtractor(args)
    elif diffusion_name == 'dit':
        return DITExtractor(args)
    else:
        raise ValueError(f"Unsupported model type: {diffusion_name}")

def main():
    args, config = parse_args_and_config()
    try:
        extractor = get_extractor(args.diffusion_name, args, config)
        print(f"[INFO] Loading {args.diffusion_name} model...")
        extractor.load_model()
        print(f"[INFO] Starting DMA extraction with {args.diffusion_name}...")
        extractor.extract_dma()
        print("[INFO] DMA extraction completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
