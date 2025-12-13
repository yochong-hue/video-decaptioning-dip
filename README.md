# Video Decaptioning with Digital Image Processing

## Abstract
Many legacy videos can only be preserved through digitization from outdated media such as film, VHS tapes, or early digital storage formats. However, the digitization process often introduces severe visual distortions, including impulse noise, compression artifacts, and sensor-related defects such as vertical colored lines. These degradations significantly complicate subtitle detection and removal, limiting the reuse of such videos for multilingual reproduction or restoration.

In this project, we propose a robust video decaptioning pipeline that combines classical Digital Image Processing (DIP) techniques with a deep video decaptioning network. Our approach focuses on improving subtitle mask extraction under degraded conditions through distortion-specific preprocessing, enabling more reliable subtitle removal without retraining the deep model. We evaluate our methods on both public benchmark datasets and self-collected videos, demonstrating that carefully designed DIP-based preprocessing can substantially improve decaptioning performance across multiple distortion types.

---

## Base Model
Our implementation builds upon the following deep learning model for video decaptioning:

- **Deep Video Decaptioning**  
  https://github.com/Linya-lab/Video_Decaptioning

We use the pretrained network provided by the authors and focus on improving the quality of subtitle masks supplied to the model through distortion-aware preprocessing.

---

## Repository Structure
Each type of distortion is handled independently using tailored preprocessing strategies.  
Please refer to the corresponding folder for detailed implementation and execution instructions.

```
video-decaptioning-dip/
├── impulse-noise/
│   ├── README.md
│   └── …
├── compression-artifacts/
│   ├── README.md
│   └── …
├── vertical-colored-lines/
│   ├── README.md
│   └── …
└── README.md
```

- **impulse_noise/**  
  Methods for handling impulse noise using median and adaptive median filtering.
- **compression_artifacts/**  
  Methods for mitigating compression-related distortions.
- **vertical_colored_lines/**  
  Methods for addressing vertical color line artifacts.

Each distortion folder contains its own README with instructions on how to run the code and reproduce results.

---

## Dataset
We evaluate our methods using the **ECCV ChaLearn 2018 LAP Video Decaptioning dataset**, which provides paired clean and subtitled videos for quantitative benchmarking.

- Dataset link:  
  https://chalearnlap.cvc.uab.cat/dataset/31/data/52/description/

This dataset is used for both testing and validation. Additional self-collected videos are also used to assess robustness under real-world degradations.

---

## Notes
- This repository focuses on preprocessing and evaluation; the deep decaptioning network itself is not retrained.
- Pretrained model weights and datasets should be downloaded separately following the instructions in the base model repository.
- Due to licensing restrictions, datasets and pretrained weights are not included in this repository.

---

## Acknowledgements
We thank the authors of *Deep Video Decaptioning* for providing the pretrained model and benchmark dataset that made this work possible.
