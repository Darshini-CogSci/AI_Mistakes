import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont 
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pathlib import Path 

# --- Import Utilities for 16-Class Classification ---
from helper.human_categories import HumanCategories, get_human_object_recognition_categories
from helper.probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
# -----------------------------------------------------

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(path):
    """Loads an image and converts it to a normalized PyTorch tensor."""
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return img, tensor

# ---------------- ROBUST MODEL LOADER ----------------
from collections import OrderedDict

def load_model(weights_path, device):
    """
    Load a ResNet-50 model from various checkpoint formats:
    - plain state_dict
    - dict with 'state_dict' or 'model' keys
    - keys possibly prefixed with 'module.' or 'model.'
    """
    model = models.resnet50(weights=None).to(device)
    
    ckpt = torch.load(weights_path, map_location="cpu")
    
    # Figure out where the actual state_dict lives
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Clean keys: strip possible 'module.' / 'model.' prefixes
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        clean_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Loaded weights from: {weights_path}")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    return model
# -----------------------------------------------------

# --- FUNCTION TO DRAW TEXT ---
def draw_text_on_image(image_array, text):
    """Draws the predicted class text onto the image array."""
    image_array = np.uint8(np.clip(image_array, 0, 255))
    pil_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text(
        (10, 10),
        text,
        fill=(0, 0, 0),
        font=font,
        stroke_fill=(255, 255, 255),
        stroke_width=2
    )
    
    return np.array(pil_img)
# -----------------------------

def get_winning_broad_category(model_outputs, decision_mapper):
    """Determines the single winning broad class name."""
    probabilities_tensor = F.softmax(model_outputs, dim=1)
    probabilities_np = probabilities_tensor.detach().cpu().numpy()[0]
    best_broad_class = decision_mapper.probabilities_to_decision(probabilities_np)
    return best_broad_class

def apply_cam_overlay(rgb_img_255, grayscale_cam):
    """
    Applies the heatmap overlay using aggressive blending (alpha=0.75)
    for vivid visualization.
    """
    # 1. Convert grayscale CAM to 0-255 BGR Jet Colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    
    # 2. Convert original image to BGR
    bgr_img_255 = cv2.cvtColor(rgb_img_255, cv2.COLOR_RGB2BGR)
    
    # 3. Blending (Aggressive Alpha: 75% Heatmap, 25% Background)
    alpha = 0.4 
    superimposed_img_bgr = cv2.addWeighted(bgr_img_255, 1.0 - alpha, heatmap, alpha, 0)
    
    # 4. Convert blended image back to RGB
    final_visualization = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)
    
    return final_visualization

def run_gradcam_on_dataset(root_input, root_output, resnet_weights, use_cuda=False):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    hc = HumanCategories()
    decision_mapper = ImageNetProbabilitiesTo16ClassesMapping(aggregation_function=np.mean)

    model = load_model(resnet_weights, device)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    Path(root_output).mkdir(parents=True, exist_ok=True)

    # ---- Walk through folder structure ----
    for root, dirs, files in os.walk(root_input):
        rel_path = os.path.relpath(root, root_input)
        out_subdir = os.path.join(root_output, rel_path)
        os.makedirs(out_subdir, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            fpath = os.path.join(root, fname)
            base_fname = os.path.splitext(fname)[0]
            
            try:
                orig_img, input_tensor = load_image(fpath)
                rgb_img_base = np.array(orig_img.resize((224, 224))).astype(np.uint8)

                if orig_img.size[0] < 16 or orig_img.size[1] < 16:
                    print(f"Skipping tiny image: {fpath}")
                    continue
            except Exception as e:
                print(f"Skipping unreadable file {fpath}. Error: {e}")
                continue

            input_tensor = input_tensor.to(device)
            
            # --- 1. Get prediction and ALL fine category indices ---
            with torch.no_grad():
                outputs = model(input_tensor)

            broad_class_name = get_winning_broad_category(outputs, decision_mapper)
            winning_indices = hc.get_imagenet_indices_for_category(broad_class_name)
            
            all_cams = []
            
            if not winning_indices:
                print(f"No fine categories found for {broad_class_name}. Skipping image.")
                continue

            # --- 2. Generate CAM for ALL fine categories and collect them ---
            for fine_idx in winning_indices:
                targets = [ClassifierOutputTarget(fine_idx)]
                individual_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                all_cams.append(individual_cam)

            # --- 3. Composite (Average) the individual heatmaps ---
            composite_cam = np.mean(np.stack(all_cams, axis=0), axis=0)

            # --- 4. Overlay and Save ---
            heatmap_resized = cv2.resize(composite_cam, (224, 224), interpolation=cv2.INTER_LINEAR)
            final_visualization = apply_cam_overlay(rgb_img_base, heatmap_resized)
            
            # Draw the text onto the visualization
            text_to_draw = f"Classified as: {broad_class_name}"
            visualization_with_text = draw_text_on_image(final_visualization, text_to_draw)

            # ---- Save result ----
            out_fname = f"{base_fname}_composite_gradcam_{broad_class_name.replace(' ', '_')}.jpg"
            out_path = os.path.join(out_subdir, out_fname)
            
            Image.fromarray(visualization_with_text).save(out_path)
            print(f"Saved COMPOSITE Heatmap for {broad_class_name}: {out_path}")
            
run_gradcam_on_dataset(
    root_input="./path/to/your/stimuli/file/",
    root_output="./path/to/your/output/file/",
    resnet_weights="./path/to/the/downloaded/resnet50/weight/",
    use_cuda=True  # or False if you want CPU
)