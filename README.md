# AI_Mistakes
A closer look at the diverse error patterns exhibited by computer vision models

This repository explores the visual competence of ResNet-50 by analyzing systematic errors and decision-making patterns. Specifically, it uses **Grad-CAM** to visualize how models prioritize texture over shape when processing manipulated stimuli to see if there is any underlying similarity in the way machine vision and human vision works.The analysis and inferences done with this codeis available in [this link](https://www.overleaf.com/read/nbrvhfnpfwwj#fbc908)

##Visual examples
<table>
  <tr>
    <td><b>Original Stimulus</b></td>
    <td><b>Grad-CAM Result (Texture Bias)</b></td>
  </tr>
  <tr>
    <td><img src="./images/original.jpg" width="300"></td>
    <td><img src="./images/heatmap.jpg" width="300"></td>
  </tr>
</table>
To use the code, you have to download the modified stimuli and helper file for16-class mapping from [Geirhos'repo](https://github.com/rgeirhos/texture-vs-shape.git) and add the directory.

## 🛠️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [[https://github.com/YOUR_USERNAME/ai-mistakes.git](https://github.com/Darshini-CogSci/AI_Mistakes.git)])
## References
-Grad-CAM: Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." (ICCV 2017).
-Library: Jacob Gildenblat's pytorch-grad-cam library.
-Dataset/Helper: Geirhos, R., et al. "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness." (ICLR 2019).
