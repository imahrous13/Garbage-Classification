# WasteWise AI: Waste Image Classification

## Overview

This is my computer vision project for classifying waste images into six categories: cardboard, glass, metal, paper, plastic, and trash. The main work is collected in a Google Colab/Jupyter notebook and covers data preparation, supervised classification, image denoising, experimental object detection and synthetic-data generation, and a small image-upload demo.

The repository also includes the dataset archive, my written report, and the presentation used to explain the project.

## Features

- Loads an image-folder dataset and applies resizing, normalization, and training-time augmentation.
- Implements a custom CNN and model factories for ResNet-18 and VGG-16 transfer learning.
- Tracks loss and accuracy, saves the best checkpoint, and can generate a classification report and confusion matrix.
- Experiments with YOLOv8, a conditional DCGAN, a denoising autoencoder, and image-text feature fusion.
- Includes a zero-shot CLIP demo with a Flask upload page and JSON prediction endpoint.
- Supports CPU or CUDA execution, with automatic mixed precision when CUDA is available.

## Tech Stack

- **Language:** Python
- **Notebook environment:** Jupyter Notebook / Google Colab
- **ML and computer vision:** PyTorch, torchvision, Ultralytics YOLOv8, Hugging Face Transformers, Pillow
- **Models:** custom CNN, ResNet-18, VGG-16, CLIP, DistilBERT, conditional DCGAN, small U-Net-style denoising autoencoder
- **Evaluation and visualization:** scikit-learn, NumPy, Matplotlib
- **Demo/API:** Flask, Waitress, TryCloudflare

## Architecture / How It Works

The notebook is organized as a set of experiments rather than one production pipeline:

1. `data.zip` is extracted and read with `torchvision.datasets.ImageFolder`.
2. Images are resized to 224×224. Training transforms add horizontal flips, rotation, and brightness jitter; validation uses resize and normalization only.
3. The classification path can train a custom CNN or replace the final layer of a pretrained ResNet-18/VGG-16. It uses cross-entropy loss, AdamW, optional cosine learning-rate scheduling, AMP, and best-checkpoint selection by validation accuracy.
4. Separate sections explore YOLOv8 detection, class-conditional image generation, and denoising with a small encoder-decoder model and skip connections.
5. The multimodal experiment concatenates ResNet-18 image features with pooled DistilBERT text features before an MLP classification head.
6. The runnable demo uses zero-shot `openai/clip-vit-base-patch32` inference. Flask accepts an uploaded image, CLIP scores six text prompts, and the UI displays the top prediction and all class probabilities.

## Technical Highlights

- Wrote reusable training and evaluation helpers with checkpointing, AMP, confusion-matrix output, and training curves.
- Computed channel statistics from the training split instead of assuming fixed normalization values.
- Built paired noisy/clean samples dynamically using blur, JPEG artifacts, Gaussian noise, and brightness/contrast changes.
- Implemented mask-aware pooling and late fusion for a two-encoder image-text classifier.
- Added both inline and offline denoising paths so the denoiser can be evaluated before a classifier or used to create a cleaned dataset copy.
- Packaged CLIP inference behind both a browser interface and a JSON endpoint served by Waitress.

## Setup

Google Colab is the intended environment because the notebook uses `/content/...` paths and notebook shell commands.

1. Open `Untitled5 (1).ipynb` in Colab.
2. Upload `data.zip` to `/content/data.zip`.
3. Run the required sections in order. The notebook installs PyTorch/torchvision, Ultralytics, and Transformers where they are first used.
4. After extraction, point `DATA_DIR` at the directory that directly contains the six class folders:

   ```python
   DATA_DIR = "/content/data/Garbage classification/Garbage classification"
   ```

5. Install any optional packages missing from the runtime:

   ```bash
   pip install scikit-learn matplotlib flask waitress pillow
   ```

A GPU is recommended for training, but the PyTorch and CLIP code selects CPU when CUDA is unavailable.

### Security note

No Gmail credential is needed to run this project. Do not reuse any email address/app-password pair that may have appeared in an earlier notebook revision. Revoke an exposed Google app password in the associated Google account and load any future secrets from Colab Secrets or environment variables instead of committing them.

## Usage

For the simplest end-to-end demo, run the final **Flask + Waitress + CLIP** notebook section. It starts a local server and attempts to create a temporary TryCloudflare URL. The API accepts multipart form data with a field named `file`:

```bash
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:7860/predict_type
```

The response contains `label`, `confidence`, and `topk`. The displayed values are CLIP softmax scores for the six prompts; they have not been verified as calibrated probabilities.

## Project Structure

```text
.
├── Untitled5 (1).ipynb                         # Training, experiments, evaluation, and web demo
├── data.zip                                    # Six-class image dataset
├── Waste Classification Project – Final Report.docx
│                                                # Detailed walkthrough and discussion
├── WasteWise-AI-Classifying-for-a-Cleaner-Planet.pptx
│                                                # Project presentation
└── README.md                                    # Repository guide
```

## Results / Metrics

These values are preserved in notebook outputs and should be read with their limitations:

- The dataset archive contains **5,054 images**: 806 cardboard, 1,002 glass, 820 metal, 1,188 paper, 964 plastic, and 274 trash images.
- The denoising autoencoder records a best average training PSNR of **28.35 dB at epoch 7**. The notebook does not contain a separate held-out denoising test result.
- A baseline CNN run records **50.4% best validation accuracy**, but that run detected two duplicated wrapper folders as classes instead of the six waste categories. It is not a valid six-class result and should not be used as a performance claim.
- The YOLO smoke test reports zero mAP. Its labels are explicitly generated as placeholders, so it verifies pipeline execution only, not detection quality.
- The multimodal run also used the incorrect two-folder class structure; its recorded accuracy is not a valid result for the intended task.

## Instructor Feedback

The final submission received **81/90** across the instructor's rubric. I am keeping this separate from the metrics above because it is an external course assessment, not a reproducible model benchmark.

- **Dataset and preprocessing — 10/10:** full credit for using the six-class TrashNet dataset and documenting resizing, normalization, augmentation, and data handling.
- **Baseline CNN and transfer learning — 14/15:** the instructor credited the custom CNN, ResNet-18, VGG-16, and zero-shot CLIP work. The main requested improvement was clearer per-class precision, recall, F1, and confusion-matrix reporting.
- **YOLO object detection — 13/15:** the YOLOv8 extension and discussion were credited, with mAP@0.5 and mAP@0.5:0.95 identified as missing from the submitted results.
- **GAN augmentation — 12/15:** implementation and qualitative samples were credited; FID/IS evaluation and ablation studies were suggested as improvements.
- **Autoencoder denoising — 12/15:** the denoising implementation and reconstructions were credited. The instructor noted missing PSNR/SSIM evaluation. The current notebook does record training PSNR, but it still has no held-out PSNR or SSIM result.
- **Multimodal extension — 5/5:** full credit for the CLIP/text-prompt extension beyond the core requirements.
- **Report, presentation, and demo — 10/10:** full credit for the written report, presentation, and Flask/Waitress/TryCloudflare demo.
- **Ethical and societal impact — 5/5:** full credit for discussing sustainability, practical deployment, and reducing sorting errors.

This feedback reflects the instructor's review of the final submission. Some positive performance comments cannot be independently reproduced from the notebook's saved outputs because of the data-root issue described below.

## Limitations / Status

This is an experimental course project and is not deployment-ready.

- The dataset has two nested folders with the same name. The current automatic directory finder stops too early, which caused several saved notebook runs to treat wrapper folders as classes. `DATA_DIR` must be set to the inner folder before retraining.
- The data split is random rather than stratified, and the notebook does not preserve split indices for exact reproduction.
- The YOLO section creates placeholder boxes from classification data. Real bounding-box annotations are required before evaluating detection.
- The GAN generated zero balancing images in the recorded run because both incorrectly detected wrapper classes had equal counts.
- No per-image text descriptions are included, so the multimodal code falls back to class-name prompts.
- The train and validation subsets in the multimodal section share one underlying dataset object; changing its transform for validation also changes the training transform.
- There are no automated tests, pinned project-wide dependency file, saved production model, input-size limit, authentication, or deployment configuration.
- TryCloudflare URLs are temporary and change whenever the demo is restarted.

## Resume Evaluation Notes

- **Strongest technical aspects:** broad hands-on PyTorch work, reusable training/evaluation code, denoising pipeline, multimodal fusion, and an inference demo.
- **Most impressive implementation:** the small U-Net-style denoising autoencoder with synthetic corruption, checkpointing by PSNR, preview grids, and inline/offline inference options.
- **Weak/basic parts:** most work is in one notebook; experiment boundaries and dependencies are not packaged; data-root handling invalidated several recorded runs.
- **Verified metrics:** 5,054 archived images with class counts listed above; 28.35 dB best recorded training PSNR for the denoiser. No valid six-class classification accuracy is recorded.
- **What is unique:** one notebook compares supervised classification, zero-shot CLIP, image-text fusion, denoising, object-detection scaffolding, and conditional generation on the same problem.
- **What may be worth highlighting on a resume:** implementation of multiple PyTorch model paths, data augmentation and normalization, AMP/checkpoint training utilities, multimodal feature fusion, and serving inference through Flask/Waitress.
- **Claims that cannot be verified from the repo:** real-world recycling impact, production usage, calibrated confidence, generalization to real waste streams, a valid six-class accuracy result, detection performance, and performance or scalability under load.
