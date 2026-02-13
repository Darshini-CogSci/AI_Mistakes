# AI_Mistakes
A closer look at the diverse error patterns exhibited by computer vision models

This repository explores the visual competence of ResNet-50 by analyzing systematic errors and decision-making patterns. Specifically, it uses **Grad-CAM** to visualize how models prioritize texture over shape when processing manipulated stimuli to see if there is any underlying similarity in the way machine vision and human vision works.

## 🚀 Key Technical Contributions
- **Composite Grad-CAM Heatmaps**: Developed a method to aggregate heatmaps across all 1000 ImageNet indices mapping to 16 broad human-centric categories (e.g., 'dog', 'clock', 'bottle'). This provides a more robust visualization of "category-level" attention.
- **Automated Research Pipeline**: A robust batch-processing script that handles directory recursion, model weight loading (DataParallel compatible), and automated labeling.
- **16-Class Mapping**: Integrated a decision-mapping layer to bridge the gap between machine-centric ImageNet classes and human-centric object recognition.

## 📂 Project Structure
- `run_gradcam.py`: The main execution script for batch-processing stimuli.
- `helper/`: Utility functions for ImageNet-to-16-category mapping (adapted from Geirhos et al.).
- `results/`: (Example outputs) Visualizations of composite heatmaps.

## 🛠️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ai-mistakes.git](https://github.com/YOUR_USERNAME/ai-mistakes.git)
