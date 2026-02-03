# Hybrid Neural Preset Generator

> **⚠️ Note:** This repository renders my previous `ai-style-transfer` project redundant. The hybrid approach implemented here significantly outperforms the old VGG-based method for photorealistic preset cloning and should be used instead.

This project implements a hybrid approach to image style transfer, specifically designed to reverse-engineer and replicate "Lightroom-style" color presets.

Unlike traditional style transfer (which often hallucinates textures or distorts edges), this tool focuses strictly on **photorealistic color grading**. It combines statistical color matching with a lightweight neural network to capture the "vibe" of a reference photo (e.g., Golden Hour, Teal & Orange, Vintage Film) and applies it to a target image without degrading its quality.

## Features

* **Hybrid Engine:** Blends **60% Statistical Color Transfer** (Reinhard algorithm) with **40% Neural Split-Toning** for robust, artifact-free results.
* **Neural Split-Toning:** Uses a custom neural network to learn separate color tints for shadows, midtones, and highlights.
* **Adaptive Film Grain:** Analyzes the noise level of the reference image and mathematically replicates the film grain on the target image, weighted towards the shadows.
* **Resolution Independent:** The color grading is learned on small previews (512px) but applied to full-resolution RAW/JPEG images instantly.
* **Structure Preservation:** Guarantees zero geometric distortion; faces and objects remain sharp.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/asechan/neural-preset.git](https://github.com/asechan/neural-preset.git)
    cd neural-preset
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install torch torchvision numpy opencv-python pillow
    ```

    *Note: If you are on macOS with Apple Silicon (M1/M2/M3), PyTorch will automatically use Metal Performance Shaders (MPS) for GPU acceleration.*

## Usage

### Quick Start
The project relies on three main scripts. To run the full hybrid pipeline:

1.  Open `hybrid_merge.py` in your code editor.
2.  Scroll to the bottom of the file to the main execution block.
3.  Update the file paths to point to your images:
    ```python
    if __name__ == "__main__":
        CONTENT = "path/to/your/photo.jpg"
        STYLE = "path/to/reference/preset.jpg"
        run_hybrid_merge(CONTENT, STYLE)
    ```
4.  Run the script:
    ```bash
    python hybrid_merge.py
    ```

The final result will be saved as `hybrid_final.jpg` in your project directory.

## Project Structure

* **`hybrid_merge.py` (Master Script):** The main entry point. It orchestrates the workflow, running both the statistical and neural engines, blending them at a 60/40 ratio, and applying the final adaptive grain.
* **`easy_preset.py`:** Implements the **Reinhard Color Transfer** algorithm. This handles the "base" color adaptation by matching the mean and standard deviation of the LAB color space.
* **`neural_preset_lite.py`:** Contains the PyTorch model for **Split-Toning**. It optimizes a parametric color filter (Shadow/Midtone/Highlight tints + Saturation/Contrast) to match the perceptual style of the reference.

## How It Works

1.  **Statistical Pass (60%):** The content image is mathematically shifted to match the global color distribution of the style reference. This provides a stable, realistic base.
2.  **Neural Pass (40%):** A lightweight CNN analyzes the image and creates a "mask" for shadows and highlights. It then optimizes specific color tints (e.g., "push teal into shadows, gold into highlights") to capture the nuanced look of the preset.
3.  **Blending:** The two results are merged. The statistical pass ensures realism, while the neural pass adds the artistic character.
4.  **Grain Synthesis:** The system measures the high-frequency noise variance in the style image and generates a matching noise field, applying it intelligently to the target image (heavier in shadows, lighter in highlights) to simulate film stock.

## License

This project is open source and available under the **MIT License**.