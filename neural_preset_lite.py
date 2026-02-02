import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class NeuralSplitTone(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable parameters for shadow, midtone, and highlight tints
        self.shadow_tint = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.midtone_tint = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.highlight_tint = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.saturation = nn.Parameter(torch.tensor([1.0]))
        self.contrast = nn.Parameter(torch.tensor([1.0]))

    def get_luminance(self, x):
        return (x[:, 0, :, :] * 0.2126) + (x[:, 1, :, :] * 0.7152) + (x[:, 2, :, :] * 0.0722)

    def forward(self, x):
        luma = self.get_luminance(x).unsqueeze(1)
        
        # Create soft masks for tonal ranges
        shadow_mask = torch.exp(-torch.pow(luma, 2) / 0.05)
        highlight_mask = torch.exp(-torch.pow(luma - 1, 2) / 0.05)
        midtone_mask = torch.clamp(1 - (shadow_mask + highlight_mask), 0, 1)

        # Apply tints
        tinted = x + (self.shadow_tint * shadow_mask) + \
                     (self.midtone_tint * midtone_mask) + \
                     (self.highlight_tint * highlight_mask)

        # Apply global contrast and saturation
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
            
            # Match mean and std dev of feature maps (color/vibe matching)
            input_mean, input_std = torch.mean(input_norm, dim=(2,3)), torch.std(input_norm, dim=(2,3))
            style_mean, style_std = torch.mean(style_norm, dim=(2,3)), torch.std(style_norm, dim=(2,3))
            
            loss += torch.mean((input_mean - style_mean) ** 2) * weights[i]
            loss += torch.mean((input_std - style_std) ** 2) * weights[i]
            
        return loss

def load_img(path, size=None):
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)

def run_neural_transfer(content_path, style_path, output_path="neural_preset_output.jpg"):
    print(f"Running neural transfer on {content_path}...")
    
    content_train = load_img(content_path, size=512)
    style_train = load_img(style_path, size=512)
    
    model = NeuralSplitTone().to(device)
    loss_fn = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    
    # Optimization loop
    for step in range(250): 
        optimizer.zero_grad()
        corrected = model(content_train)
        loss = loss_fn(corrected, style_train) * 1000
        loss.backward()
        optimizer.step()
        
    # Apply to full resolution image
    full_content = load_img(content_path)
    with torch.no_grad():
        final_result = model(full_content)
    
    save_transform = transforms.ToPILImage()
    img_out = save_transform(final_result.squeeze().cpu())
    img_out.save(output_path)
    return output_path

if __name__ == "__main__":
    CONTENT = "WhatsApp Image 2025-01-21 at 19.24.33.jpeg"
    STYLE = "DSC_1.png"
    run_neural_transfer(CONTENT, STYLE)