# Swarm Microrobots AR Semantic Segmentation of Cracks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation for "HeSARIC: A Heterogeneous Cyber–Physical Robotic Swarm Framework for Structural Health Monitoring with Augmented Reality Representation" published in [MDPI Micromachines](https://www.mdpi.com/2072-666X/16/4/460). Our approach leverages a swarm of microrobots equipped with cameras to detect cracks in narrow spaces, using semantic segmentation and augmented reality for visualization and control.

![System Overview](https://www.mdpi.com/micromachines/micromachines-16-00460/article_deploy/html/images/micromachines-16-00460-g018-550.jpg)

## Features

- **Semantic Segmentation Model**: UNet architecture optimized for crack detection of confined spaces
- **Multi-Camera Integration**: Processes inputs from multiple microrobots simultaneously
- **Augmented Reality Representation**: Visualizes detected crack masks
- **Robust to Environmental Variations**: Works under various lighting and perspective conditions

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/alireza-fath/Swarm-Microrobots-AR-Semantic-Segmentation-of-Cracks.git
cd Swarm-Microrobots-AR-Semantic-Segmentation-of-Cracks

# Install dependencies
pip install torch torchvision opencv-python numpy matplotlib scipy pillow requests
```

### Model Weights

**Important**: The model weights file is too large to be included directly in the repository. You need to download it separately:

1. Go to the [Releases page](https://github.com/alireza-fath/Swarm-Microrobots-AR-Semantic-Segmentation-of-Cracks/releases)
2. Download the `finetuned-all-occlude.pt` file from the latest release
3. Place it in the `weights` folder in the root directory

```
project-root/
├── aug-model/
├── weights/
│   └── finetuned-all-occlude.pt  # <- Place downloaded weights here
├── README.md
...
```

## Usage

The main functionality is demonstrated in the Jupyter notebook `aug-model/Crack-Semantic-Segmentation-of-Swarm-of-Microrobots-and-Merging.ipynb`.

### Basic Usage Example

```python
import torch
from crackseg.models import UNet
from PIL import Image
import numpy as np
from inference import preprocess

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("weights/finetuned-all-occlude.pt", map_location=device)
model = UNet(in_channels=3, out_channels=2)
model.load_state_dict(checkpoint["model"].float().state_dict())
model.eval()
model.to(device)

# Load and preprocess image
input_image = Image.open("path/to/image.jpg")
tensor_image = preprocess(input_image).unsqueeze(0).to(device)

# Inference
SENSITIVITY = -10  # Adjust threshold for crack detection
with torch.no_grad():
    output = model(tensor_image).cpu()
    output = output[:,0,:,:] < output[:,1,:,:] + SENSITIVITY

# Get mask
mask = output[0].long().squeeze().numpy()
```

### Multi-Camera Integration

The system can process inputs from multiple ESP32-CAM microrobots simultaneously:

```python
import requests
from PIL import Image
from io import BytesIO

# Example for capturing from multiple microrobots
cameras = [
    "http://192.168.137.96/capture",
    "http://192.168.137.147/capture",
    "http://192.168.137.253/capture"
]

images = []
for camera_url in cameras:
    response = requests.get(camera_url)
    image = Image.open(BytesIO(response.content))
    images.append(image)
    
# Process each image through the model...
```

## Methodology

Our approach uses a modified UNet architecture for semantic segmentation of cracks, fine-tuned with extensive data augmentation to handle real-world variations:

1. **Data Collection**: Multiple ESP32-CAM microrobots capture images from different angles
2. **Data Processing**: Images undergo augmentation (perspective transformation, gradients, random crops)
3. **Semantic Segmentation**: UNet model identifies crack regions at pixel level
4. **Multi-View Integration**: Results from multiple robots are merged using weighted averaging
5. **AR Visualization**: Detected cracks are highlighted in an augmented reality interface

## Contributors

This work was developed by:
Alireza Fath [1], Christoph Sauter [1], Yi Liu [1], Brandon Gamble [1], Dylan Burns [1], Evan Trombley [1], Sai Krishna Reddy Sathi [1,2], Tian Xia [3], and Dryver Huston* [1].

1. Department of Mechanical Engineering, University of Vermont
2. Department of Mechanical Engineering, Indian Institute of Technology Madras
3. Department of Electrical and Biomedical Engineering, University of Vermont
*Author to whom correspondence should be addressed.


## Citation

If you use this code or method in your research, please cite our paper:

```
Fath, A.; Sauter, C.; Liu, Y.; Gamble, B.; Burns, D.; Trombley, E.; Sathi, S.K.R.; Xia, T.; Huston, D. HeSARIC: A Heterogeneous Cyber–Physical Robotic Swarm Framework for Structural Health Monitoring with Augmented Reality Representation. Micromachines 2025, 16, 460. https://doi.org/10.3390/mi16040460
```

## Previous Models Used in the development

- The UNet implementation is built upon the work by [Yakhyokhuja Valikhujaev](https://github.com/yakupov/crackseg)
- Ronneberger, O.; Fischer, P.; Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Proceedings of the Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, 5–9 October 2015; Proceedings, part III 18. Springer: Berlin/Heidelberg, Germany, 2015; pp. 234–241. 
- Ha, K. Crack_Segmentation. Available online: https://github.com/khanhha/crack_segmentation (accessed on 1 April 2024).
- Zhang, L.; Yang, F.; Zhang, Y.D.; Zhu, Y.J. Road crack detection using deep convolutional neural network. In Proceedings of the 2016 IEEE International Conference on Image Processing (ICIP), Phoenix, AZ, USA, 25–28 September 2016; IEEE: Piscataway, NJ, USA, 2016; pp. 3708–3712. 
- Yang, F.; Zhang, L.; Yu, S.; Prokhorov, D.; Mei, X.; Ling, H. Feature pyramid and hierarchical boosting network for pavement crack detection. IEEE Trans. Intell. Transp. Syst. 2019, 21, 1525–1535. 
- Eisenbach, M.; Stricker, R.; Seichter, D.; Amende, K.; Debes, K.; Sesselmann, M.; Ebersbach, D.; Stoeckert, U.; Gross, H.-M. How to get pavement distress detection ready for deep learning? In A systematic approach. In Proceedings of the 2017 International Joint Conference on Neural Networks (IJCNN), Anchorage, AK, USA, 14–19 May 2017; IEEE: Piscataway, NJ, USA, 2017; pp. 2039–2047. 
- Shi, Y.; Cui, L.; Qi, Z.; Meng, F.; Chen, Z. Automatic road crack detection using random structured forests. IEEE Trans. Intell. Transp. Syst. 2016, 17, 3434–3445. 
- Amhaz, R.; Chambon, S.; Idier, J.; Baltazart, V. Automatic crack detection on two-dimensional pavement images: An algorithm based on minimal path selection. IEEE Trans. Intell. Transp. Syst. 2016, 17, 2718–2729. 
- Zou, Q.; Cao, Y.; Li, Q.; Mao, Q.; Wang, S. CrackTree: Automatic crack detection from pavement images. Pattern Recognit. Lett. 2012, 33, 227–238. 
- Aidonchuk, A. Cracks Segmentation Dataset. Available online: https://github.com/aidonchuk/cracks_segmentation_dataset (accessed on 25 June 2024).
- Leo, Y. DeepCrack. Available online: https://github.com/yhlleo/DeepCrack (accessed on 25 June 2024).
- Lab, C.R. CCNY Robotics Lab. Available online: https://github.com/CCNYRoboticsLab/concreteIn_inpection_VGGF (accessed on 25 June 2024).
- Özgenel, Ç.F. Concrete crack segmentation dataset. Mendeley Data 2019, 1, 2019. [Google Scholar]
- Wang, K.; Fang, B.; Qian, J.; Yang, S.; Zhou, X.; Zhou, J. Perspective transformation data augmentation for object detection. IEEE Access 2019, 8, 4935–4943. [Google Scholar] [CrossRef]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The original model used for fine-tuning is also under MIT License - see [Previous Model LICENSE](aug-model/Previous%20Model%20LICENSE).
