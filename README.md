# AI_Mistakes
A closer look at the diverse error patterns exhibited by computer vision models

This repository explores the visual competence of ResNet-50 by analyzing systematic errors and decision-making patterns. Specifically, it uses **Grad-CAM** to visualize how models prioritize texture over shape when processing manipulated stimuli to see if there is any underlying similarity in the way machine vision and human vision works.The analysis and inferences done with this codeis available in [this link](https://www.overleaf.com/read/nbrvhfnpfwwj#fbc908)

##Visual examples
<table>
  <tr>
    <td><b>Original Stimulus (Edges)</b></td>
    <td><b>Grad-CAM Result </b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/afe968fc-d88c-4132-aea8-de6aa0ddaaa8" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/8f530e4b-f5dc-4530-aa9c-11401ade36f4" width="300"></td>
  </tr>
</table>
<table>
  <tr>
    <td><b>Original Stimulus (Filled-silhouette)</b></td>
    <td><b>Grad-CAM Result (Same label, different focus)</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7bf936e4-4c9d-4de5-9771-d3cf5da5c583" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/a02aa422-313f-431b-af74-341dafa4bdcc" width="300"></td>
  </tr>
</table>
</table>
<table>
  <tr>
    <td><b>Original Stimulus (Style-preprocessed)</b></td>
    <td><b>Grad-CAM Result (Texture bias focus)</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1498a968-5b42-47b0-a69f-d174da869988" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/70d74f94-21c5-475f-aa7e-c282e90057fd" width="300"></td>
  </tr>
</table>
To use the code, you have to download the modified stimuli and helper file for16-class mapping from [Geirhos'repo](https://github.com/rgeirhos/texture-vs-shape.git) and add to this directory.

## 🛠️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [[https://github.com/YOUR_USERNAME/ai-mistakes.git](https://github.com/Darshini-CogSci/AI_Mistakes.git)])
## References
-Grad-CAM: Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." (ICCV 2017).

-Library: Jacob Gildenblat's pytorch-grad-cam library.

-Dataset/Helper: Geirhos, R., et al. "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness." (ICLR 2019).
