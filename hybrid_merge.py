import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------
# Part 1: Statistical Transfer Logic
# ---------------------------------------------------------
def run_statistical_transfer(source, target):
    # Convert PIL to numpy for OpenCV
    source = np.array(source)
    target = np.array(target)
    
    source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, l_std_src) = (source_lab[:,:,0].mean(), source_lab[:,:,0].std())
    (a_mean_src, a_std_src) = (source_lab[:,:,1].mean(), source_lab[:,:,1].std())
    (b_mean_src, b_std_src) = (source_lab[:,:,2].mean(), source_lab[:,:,2].std())

    (l_mean_tar, l_std_tar) = (target_lab[:,:,0].mean(), target_lab[:,:,0].std())
    (a_mean_tar, a_std_tar) = (target_lab[:,:,1].mean(), target_lab[:,:,1].std())
    (b_mean_tar, b_std_tar) = (target_lab[:,:,2].mean(), target_lab[:,:,2].std())

    # Align target statistics to source statistics
    target_lab[:,:,0] -= l_mean_tar
    target_lab[:,:,1] -= a_mean_tar
    target_lab[:,:,2] -= b_mean_tar

    target_lab[:,:,0] = (target_lab[:,:,0] * (l_std_src / l_std_tar))
    target_lab[:,:,1] = (target_lab[:,:,1] * (a_std_src / a_std_tar))
    target_lab[:,:,2] = (target_lab[:,:,2] * (b_std_src / b_std_tar))

    target_lab[:,:,0] += l_mean_src
    target_lab[:,:,1] += a_mean_src
    target_lab[:,:,2] += b_mean_src

    target_lab = np.clip(target_lab, 0, 255).astype("uint8")
    result_bgr = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

# ---------------------------------------------------------
# Part 2: Neural Transfer Logic
# ---------------------------------------------------------
class NeuralSplitTone(nn.Module):
    def __init__(self):
        super().__init__()
        self.shadow_tint = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.midtone_tint = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.highlight_tint = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.saturation = nn.Parameter(torch.tensor([1.0]))
        self.contrast = nn.Parameter(torch.tensor([1.0]))

    def get_luminance(self, x):
        return (x[:, 0, :, :] * 0.2126) + (x[:, 1, :, :] * 0.7152) + (x[:, 2, :, :] * 0.0722)

    def forward(self, x):
        luma = self.get_luminance(x).unsqueeze(1)
        shadow_mask = torch.exp(-torch.pow(luma, 2) / 0.05)
        highlight_mask = torch.exp(-torch.pow(luma - 1, 2) / 0.05)
        midtone_mask = torch.clamp(1 - (shadow_mask + highlight_mask), 0, 1)

        tinted = x + (self.shadow_tint * shadow_mask) + \
                     (self.midtone_tint * midtone_mask) + \
                     (self.highlight_tint * highlight_mask)

        tinted = (tinted - 0.5) * self.contrast + 0.5
        luma_tinted = self.get_luminance(tinted).unsqueeze(1)
        tinted = luma_tinted + (tinted - luma_tinted) * self.saturation
        return torch.clamp(tinted, 0, 1)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),
            nn.Sequential(*list(vgg.children())[4:9]),
            nn.Sequential(*list(vgg.children())[9:16])
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    def forward(self, input_img, style_img):
        input_norm = (input_img - self.mean) / self.std
        style_norm = (style_img - self.mean) / self.std
        loss = 0
        weights = [1.0, 0.5, 0.1]
        for i, slice_block in enumerate(self.slices):
            input_norm = slice_block(input_norm)
            style_norm = slice_block(style_norm)
            input_mean, input_std = torch.mean(input_norm, dim=(2,3)), torch.std(input_norm, dim=(2,3))
            style_mean, style_std = torch.mean(style_norm, dim=(2,3)), torch.std(style_norm, dim=(2,3))
            loss += torch.mean((input_mean - style_mean) ** 2) * weights[i]
            loss += torch.mean((input_std - style_std) ** 2) * weights[i]
        return loss

def load_img(img_pil, size=None):
    if size:
        img_pil = img_pil.resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img_pil).unsqueeze(0).to(device)

def run_neural_optimization(content_pil, style_pil):
    content_train = load_img(content_pil, size=512)
    style_train = load_img(style_pil, size=512)
    
    model = NeuralSplitTone().to(device)
    loss_fn = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    
    print("Optimizing neural model...")
    for step in range(200):
        optimizer.zero_grad()
        corrected = model(content_train)
        loss = loss_fn(corrected, style_train) * 1000
        loss.backward()
        optimizer.step()
        
    full_content = load_img(content_pil)
    with torch.no_grad():
        final_result = model(full_content)
    
    return transforms.ToPILImage()(final_result.squeeze().cpu())

# ---------------------------------------------------------
# Part 3: Adaptive Grain Logic
# ---------------------------------------------------------
def apply_adaptive_grain(content_pil, style_pil, strength=1.0):
    print("Applying adaptive film grain...")
    content_tensor = transforms.ToTensor()(content_pil).unsqueeze(0).to(device)
    style_tensor = transforms.ToTensor()(style_pil).unsqueeze(0).to(device)
    
    # Calculate luminance of style image
    style_gray = 0.299*style_tensor[:,0,:,:] + 0.587*style_tensor[:,1,:,:] + 0.114*style_tensor[:,2,:,:]
    style_gray = style_gray.unsqueeze(1)
    
    # Estimate grain intensity by comparing image to a blurred version
    blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=1.5)
    blurred_style = blurrer(style_gray)
    noise_map = style_gray - blurred_style
    grain_intensity = torch.std(noise_map) * strength * 2.5 
    
    # Generate Gaussian noise
    grain = torch.randn_like(content_tensor) * grain_intensity
    
    # Create mask to apply grain mostly to shadows and midtones
    content_luma = 0.299*content_tensor[:,0,:,:] + 0.587*content_tensor[:,1,:,:] + 0.114*content_tensor[:,2,:,:]
    content_luma = content_luma.unsqueeze(1)
    grain_mask = torch.clamp((1.0 - content_luma) * 1.2, 0, 1)
    
    final_img = content_tensor + (grain * grain_mask)
    return transforms.ToPILImage()(final_img.squeeze().cpu().clamp(0,1))

# ---------------------------------------------------------
# Part 4: Main Execution
# ---------------------------------------------------------
def run_hybrid_merge(content_path, style_path, output_path="hybrid_final.jpg"):
    print(f"Starting hybrid preset merge for {content_path}")
    
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    
    # 1. Run Statistical Transfer
    print("Step 1: Running statistical transfer...")
    stat_result = run_statistical_transfer(style_img, content_img)
    
    # 2. Run Neural Transfer
    print("Step 2: Running neural transfer...")
    neural_result = run_neural_optimization(content_img, style_img)
    
    # 3. Blend Results
    print("Step 3: Blending results (60% Statistical / 40% Neural)...")
    if neural_result.size != stat_result.size:
        neural_result = neural_result.resize(stat_result.size, Image.LANCZOS)
    
    # Blend formula: out = image1 * (1.0 - alpha) + image2 * alpha
    # We want 60% Statistical and 40% Neural.
    # Therefore alpha should be 0.4.
    merged = Image.blend(stat_result, neural_result, alpha=0.4)
    
    # 4. Apply Grain
    print("Step 4: Applying grain...")
    final_result = apply_adaptive_grain(merged, style_img, strength=1.2)
    
    final_result.save(output_path, quality=95)
    print(f"Success! Saved result to {output_path}")

if __name__ == "__main__":
    CONTENT = "WhatsApp Image 2025-01-21 at 19.24.33.jpeg"
    STYLE = "DSC_1.png"
    run_hybrid_merge(CONTENT, STYLE)