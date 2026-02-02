import cv2
import numpy as np
import os

def run_statistical_transfer(source_path, target_path, output_path="easy_preset_output.jpg"):
    """
    Applies Reinhard color transfer to match the color distribution of the source 
    image to the target image in LAB color space.
    """
    print(f"Running statistical transfer on {source_path}...")
    
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    
    if source is None or target is None:
        print("Error: Could not read images.")
        return None

    # Convert to LAB color space to separate luminance from color channels
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Calculate mean and standard deviation for source and target
    (l_mean_src, l_std_src) = (source_lab[:,:,0].mean(), source_lab[:,:,0].std())
    (a_mean_src, a_std_src) = (source_lab[:,:,1].mean(), source_lab[:,:,1].std())
    (b_mean_src, b_std_src) = (source_lab[:,:,2].mean(), source_lab[:,:,2].std())

    (l_mean_tar, l_std_tar) = (target_lab[:,:,0].mean(), target_lab[:,:,0].std())
    (a_mean_tar, a_std_tar) = (target_lab[:,:,1].mean(), target_lab[:,:,1].std())
    (b_mean_tar, b_std_tar) = (target_lab[:,:,2].mean(), target_lab[:,:,2].std())

    # Subtract target mean
    target_lab[:,:,0] -= l_mean_tar
    target_lab[:,:,1] -= a_mean_tar
    target_lab[:,:,2] -= b_mean_tar

    # Scale by the standard deviation ratio
    target_lab[:,:,0] = (target_lab[:,:,0] * (l_std_src / l_std_tar))
    target_lab[:,:,1] = (target_lab[:,:,1] * (a_std_src / a_std_tar))
    target_lab[:,:,2] = (target_lab[:,:,2] * (b_std_src / b_std_tar))

    # Add source mean
    target_lab[:,:,0] += l_mean_src
    target_lab[:,:,1] += a_mean_src
    target_lab[:,:,2] += b_mean_src

    # Clip values to valid range and convert back to BGR
    target_lab = np.clip(target_lab, 0, 255).astype("uint8")
    result = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(output_path, result)
    return result

if __name__ == "__main__":
    # Update paths as needed
    CONTENT_FILE = "WhatsApp Image 2025-01-21 at 19.24.33.jpeg"
    STYLE_FILE = "DSC_1.png"
    run_statistical_transfer(STYLE_FILE, CONTENT_FILE)