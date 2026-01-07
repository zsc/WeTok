import argparse
import os
import sys
import json
import base64
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import importlib
from einops import rearrange

# Add current directory to path so src.WeTok can be imported
sys.path.append(os.getcwd())

# Define device
if hasattr(torch, "npu") and torch.npu.is_available():
    DEVICE = torch.device("npu:0")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model(config, ckpt_path=None):
    if "model" in config and "class_path" in config.model:
        model_cls = get_obj_from_str(config.model.class_path)
    else:
        # Fallback to the class used in reconstruct_image.py
        from src.WeTok.models.lfqgan import VQModel
        model_cls = VQModel
        
    model = model_cls(**config.model.init_args)
    
    if ckpt_path is not None:
        print(f"Loading checkpoint from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        # Handle potential prefix mismatches if necessary
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
    model.eval()
    return model

def preprocess_image(image_path, size):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    
    image = image.resize((size, size), Image.BICUBIC)
    
    x = np.array(image).astype(np.float32)
    x = (x / 127.5) - 1.0 # Normalize to [-1, 1]
    x = torch.from_numpy(x).permute(2, 0, 1) # (C, H, W)
    x = x.unsqueeze(0) # (B, C, H, W)
    return x

def encode_indices(indices):
    """
    Convert indices tensor (or list of tensors) to base64 string(s).
    """
    if isinstance(indices, (list, tuple)):
        encoded_list = []
        for idx in indices:
            arr = idx.detach().cpu().numpy()
            if arr.max() < 256:
                arr = arr.astype(np.uint8)
            elif arr.max() < 65536:
                arr = arr.astype(np.uint16)
            else:
                arr = arr.astype(np.int64)
            
            b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
            encoded_list.append({
                "data": b64,
                "dtype": str(arr.dtype),
                "shape": arr.shape
            })
        return encoded_list
    else:
        arr = indices.detach().cpu().numpy()
        if arr.max() < 256:
            arr = arr.astype(np.uint8)
        elif arr.max() < 65536:
            arr = arr.astype(np.uint16)
        else:
            arr = arr.astype(np.int64)
            
        b64 = base64.b64encode(arr.tobytes()).decode('utf-8')
        return {
            "data": b64,
            "dtype": str(arr.dtype),
            "shape": arr.shape
        }

def decode_indices(encoded_data):
    if isinstance(encoded_data, list):
        indices = []
        for d in encoded_data:
            data = base64.b64decode(d['data'])
            arr = np.frombuffer(data, dtype=d['dtype']).reshape(d['shape'])
            indices.append(torch.from_numpy(arr))
        return indices
    else:
        data = base64.b64decode(encoded_data['data'])
        arr = np.frombuffer(data, dtype=encoded_data['dtype']).reshape(encoded_data['shape'])
        return torch.from_numpy(arr.copy())

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def main():
    parser = argparse.ArgumentParser(description="Generate WeTok from image or Reconstruct from WeTok")
    
    # Mode argument
    parser.add_argument("--mode", choices=["encode", "decode", "auto"], default="auto", help="Operation mode: encode (image->json), decode (json->image) or auto (detect from extension)")
    
    # Positional Input/Output
    parser.add_argument("input", type=str, help="Path to input file (Image for encode, JSON for decode)")
    parser.add_argument("output", type=str, help="Path to output file (JSON for encode, Image for decode)")
    
    # Shared / Mode specific
    parser.add_argument("--size", default=256, type=int, help="Image input size (encode mode)")
    parser.add_argument("--config", type=str, help="Path to model config (yaml). Required for encode, optional for decode")
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint. Required for encode, optional for decode")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
        
    # Auto-detect mode
    mode = args.mode
    if mode == "auto":
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.json']:
            mode = "decode"
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.avif']:
            mode = "encode"
        else:
            print(f"Error: Could not automatically determine mode from extension '{ext}'. Please specify --mode.")
            sys.exit(1)
            
    print(f"Operation mode: {mode}")
    
    if mode == "encode":
        if not args.config or not args.ckpt:
            print("Error: --config and --ckpt are required for encode mode.")
            sys.exit(1)
            
        if not os.path.exists(input_path):
            print(f"Error: Input image not found at {input_path}")
            sys.exit(1)
            
        # Load Config
        config = OmegaConf.load(args.config)
        
        # Load Model
        model = load_model(config, args.ckpt).to(DEVICE)
        
        # Preprocess Image
        img_tensor = preprocess_image(input_path, args.size).to(DEVICE)
        
        # Encode
        print("Encoding...")
        with torch.no_grad():
            if model.use_ema:
                with model.ema_scope():
                    quant, diff, indices, _ = model.encode(img_tensor)
            else:
                quant, diff, indices, _ = model.encode(img_tensor)
                
        # Serialize
        wetok_data = encode_indices(indices)
        
        output_data = {
            "wetok": wetok_data,
            "config": args.config,
            "ckpt": args.ckpt,
            "image_size": args.size
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"Successfully saved WeTok to {args.output}")

    elif mode == "decode":
        if not os.path.exists(input_path):
             print(f"Error: Input JSON not found at {input_path}")
             sys.exit(1)

        with open(input_path, 'r') as f:
            data = json.load(f)
        
        wetok_data = data['wetok']
        config_path = args.config if args.config else data['config']
        ckpt_path = args.ckpt if args.ckpt else data['ckpt']
        image_size = data['image_size']
        
        print(f"Loading config from {config_path}")
        if not os.path.exists(config_path):
             if os.path.exists(os.path.join("../WeTok", config_path)):
                 config_path = os.path.join("../WeTok", config_path)
             elif os.path.exists(os.path.join("..", config_path)):
                 config_path = os.path.join("..", config_path)
        
        config = OmegaConf.load(config_path)
        
        if not os.path.exists(ckpt_path):
             if os.path.exists(os.path.join("../WeTok", ckpt_path)):
                 ckpt_path = os.path.join("../WeTok", ckpt_path)
             elif os.path.exists(os.path.join("..", ckpt_path)):
                 ckpt_path = os.path.join("..", ckpt_path)

        model = load_model(config, ckpt_path).to(DEVICE)
        
        indices = decode_indices(wetok_data)
        
        # Move to device
        if isinstance(indices, list):
            indices = [idx.to(DEVICE) for idx in indices]
        else:
            indices = indices.to(DEVICE)
            
        num_codebooks = config.model.init_args.get('num_codebooks', 1)
        
        ch_mult = config.model.init_args.ddconfig.ch_mult
        downsample_factor = 2 ** (len(ch_mult) - 1)
        res = image_size // downsample_factor
        
        print(f"Reconstructing with resolution {res}x{res}, num_codebooks {num_codebooks}")

        with torch.no_grad():
            if isinstance(indices, (list, tuple)):
                print("Error: Token factorization reconstruction not yet implemented in this simple script.")
                sys.exit(1)
            else:
                indices = indices.view(1, res * res, num_codebooks)
                
                if model.use_ema:
                    with model.ema_scope():
                        quant = model.quantize.decode(indices)
                else:
                    quant = model.quantize.decode(indices)
                    
                quant = rearrange(quant, 'b (h w) d -> b d h w', h=res, w=res)
                
                reconstructed_images = model.decode(quant)
                
        img = custom_to_pil(reconstructed_images[0])
        img.save(args.output)
        print(f"Successfully reconstructed image to {args.output}")

if __name__ == "__main__":
    main()
