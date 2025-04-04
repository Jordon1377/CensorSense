# CensorSense: Smart Facial Censorship

**Project Leads:**  
Faizaan Ali ([faizaan5ali](https://github.com/faizaan5ali))  
Jordon Rolley ([Jordon1377](https://github.com/Jordon1377))  

**Repository:** [CensorSense on GitHub](https://github.com/Jordon1377/CensorSense)  
**License:** MIT License

---

## Overview

CensorSense is an AI-powered application that enables selective facial censorship in video. Users can choose specific individuals to be censored, and our model will track and pixelate their faces throughout the video. This project is designed to support privacy in digital content and is especially useful for journalists, researchers, and content creators.

This project utilizes the **Multi-task Cascaded Convolutional Networks (MTCNN)** architecture for robust face detection and alignment and is trained using the **WIDER FACE dataset**, a standard benchmark for face detection in the wild.

---

## Face Detection with MTCNN

We implemented face detection and alignment using the architecture described in:

> K. Zhang, Z. Zhang, Z. Li and Y. Qiao, "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," *IEEE Signal Processing Letters*, vol. 23, no. 10, pp. 1499-1503, Oct. 2016. [DOI](https://doi.org/10.1109/LSP.2016.2603342)

### Why MTCNN?

- **Joint Detection and Alignment**: Simultaneously detects face bounding boxes and facial landmarks.
- **Robust Performance**: Performs well under challenging conditions (e.g., occlusion, lighting variations).
- **Efficiency**: Lightweight and suitable for integration into real-time or near real-time video processing pipelines.

MTCNN consists of three stages:
1. **P-Net** (Proposal Network) – generates candidate windows.
2. **R-Net** (Refine Network) – filters and refines results from P-Net.
3. **O-Net** (Output Network) – produces final bounding boxes and facial landmarks.

---

## Training Data: WIDER FACE Dataset

Our models were trained and evaluated using the [WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/), a benchmark dataset for face detection tasks.

> S. Yang, P. Luo, C. C. Loy and X. Tang, "WIDER FACE: A Face Detection Benchmark," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 5525–5533. [DOI](https://doi.org/10.1109/CVPR.2016.596)

### Features of WIDER FACE:
- **32,203 images and 393,703 labeled faces** in total.
- Contains a wide range of challenges including pose variations, occlusions, and complex backgrounds.
- Widely used to benchmark state-of-the-art face detection algorithms.

---

## Setup

> Coming soon — instructions on installing dependencies and running the face detection model locally.

---

## Goals

- Create and train a robust model to detect faces and apply corresponding bounding boxes
- Use NMS regression to remove extraneous boxes and refine relevant ones 
- Apply pixelation over time to track individuals throughout the video.
- Build a basic UI to support user input and visualization.

---

## Acknowledgements

- MTCNN authors for releasing their research and architecture.
- WIDER FACE dataset contributors for their benchmark dataset.
- Thanks to our instructors and peers for ongoing feedback.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
