import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import logging
import os
import threading

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512 # A reasonable size for web processing
VGG_WEIGHTS_PATH = './model_data/vgg19-dcbb9e9d.pth'
YOLO_MODEL_PATH = 'yolov8l-seg.pt'

log = logging.getLogger(__name__)

# --- Models are now initially None ---
yolo_model = None
vgg_features = None

# --- A lock for thread-safe model loading ---
nst_lock = threading.Lock()


# --- Image Utilities ---
def load_and_resize(path, size=IMG_SIZE):
    try:
        img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
        return img
    except Exception as e:
        log.error(f"Error loading image {path}: {e}")
        return None

def pil_to_tensor(img):
    return transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

def tensor_to_pil(tensor):
    t = tensor.clone().detach().cpu().squeeze(0)
    t.clamp_(0, 1)
    return transforms.ToPILImage()(t)

# --- Neural Network and Loss Functions ---
def get_features(x, model, layers):
    feats = []
    for idx, layer in enumerate(model):
        x = layer(x)
        if str(idx) in layers:
            feats.append(x)
    return feats

def gram_matrix(t):
    b, c, h, w = t.size()
    f = t.view(b, c, h * w)
    # Using torch.bmm for batch matrix multiplication
    return torch.bmm(f, f.transpose(1, 2)) / (h * w)

def resize_mask(mask, target_tensor):
    return F.interpolate(mask, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])) + \
           torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]))

def blur_mask(mask_tensor, sigma=5.0):
    if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
        arr = mask_tensor.squeeze().cpu().numpy()
        arr_blur = cv2.GaussianBlur(arr, (0, 0), sigma)
        blur = torch.from_numpy(arr_blur).float().clamp(0, 1).to(mask_tensor.device)
        return blur.unsqueeze(0).unsqueeze(0)
    return mask_tensor

# --- On-Demand Model Loader ---
def _load_nst_models():
    """Internal function to load YOLO and VGG models. Should not be called directly."""
    global yolo_model, vgg_features
    log.info("Loading NST models (YOLO, VGG19) into memory for the first time...")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    vgg = models.vgg19(weights=None)
    vgg.load_state_dict(torch.load(VGG_WEIGHTS_PATH, map_location=DEVICE))
    vgg_features = vgg.features.to(DEVICE).eval()
    
    log.info("NST models loaded successfully.")

def ensure_nst_models_loaded():
    """
    Checks if NST models are loaded. If not, loads them in a thread-safe way.
    This is the function that should be called before running the main process.
    """
    global yolo_model, vgg_features
    if yolo_model is None or vgg_features is None:
        with nst_lock:
            # Double-check inside the lock to prevent race conditions
            if yolo_model is None or vgg_features is None:
                _load_nst_models()

# --- Main Style Transfer Function ---
def run_style_transfer(content_path, style_path, output_path, update_callback=None):
    """
    Main function to execute the entire style transfer process.
    It ensures models are loaded before proceeding.
    """
    # This is the crucial call to the on-demand loader
    ensure_nst_models_loaded()
    
    log.info(f"Starting style transfer. Content: {content_path}, Style: {style_path}")
    log.info(f"Using device: {DEVICE}")

    # 1. Load Images
    cloth_img = load_and_resize(content_path)
    style_img = load_and_resize(style_path)
    if cloth_img is None or style_img is None:
        raise ValueError("Could not load content or style image.")

    content_tensor = pil_to_tensor(cloth_img)
    style_tensor = pil_to_tensor(style_img)

    # 2. YOLOv8 Segmentation Mask
    if update_callback: update_callback('Detecting clothing item...')
    log.info("Running YOLO segmentation...")
    results = yolo_model(cloth_img, imgsz=IMG_SIZE, conf=0.3, verbose=False)

    mask = None
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        log.info("Object detected. Creating soft mask.")
        mask_np = results[0].masks.data.cpu().numpy()[0]
        mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        mask = blur_mask(mask, sigma=7.0)
    else:
        log.warning("No object detected by YOLO. Applying style to entire image.")
        mask = torch.ones_like(content_tensor[:, :1, :, :])

    # 3. VGG19 Feature Extraction
    if update_callback: update_callback('Setting up neural network...')
    log.info("Extracting VGG19 features...")
    content_layers = ["21"]
    style_layers = ["0", "5", "10", "19", "28"]

    with torch.no_grad():
        content_feat = get_features(content_tensor, vgg_features, content_layers)[0]
        style_feats = get_features(style_tensor, vgg_features, style_layers)
        style_grams = [gram_matrix(f) for f in style_feats]

    # 4. Style Transfer Optimization Loop
    log.info("Starting optimization loop...")
    input_img = content_tensor.clone().requires_grad_(True)
    optimizer = optim.Adam([input_img], lr=0.03)
    num_steps = 300
    alpha, beta, gamma = 1e1, 1e6, 1e-6

    for step in range(num_steps):
        gen_style_feats = get_features(input_img, vgg_features, style_layers)
        gen_content_feat = get_features(input_img, vgg_features, content_layers)[0]

        c_mask = resize_mask(mask, gen_content_feat)
        content_loss = alpha * F.mse_loss(gen_content_feat * c_mask, content_feat * c_mask)

        style_loss = 0
        for f, g in zip(gen_style_feats, style_grams):
            s_mask = resize_mask(mask, f)
            style_loss += F.mse_loss(gram_matrix(f * s_mask), g)
        style_loss *= beta
        
        tv_loss = gamma * total_variation_loss(input_img)
        total_loss = content_loss + style_loss + tv_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            progress_msg = f'Styling in progress: Step {step+1}/{num_steps}'
            log.info(progress_msg)
            if update_callback: update_callback(progress_msg)
    
    # 5. Masked Blending and Saving Result
    log.info("Optimization finished. Blending and saving final image.")
    with torch.no_grad():
        input_img.clamp_(0, 1)
        stylized = input_img.detach()
        mask_resz = resize_mask(mask, stylized).expand(-1, 3, -1, -1)
        blended = stylized * mask_resz + content_tensor * (1 - mask_resz)

    result_pil = tensor_to_pil(blended)
    result_pil.save(output_path)
    log.info(f"Successfully saved stylized image to {output_path}")

    return True