[![Open In Kaggle](https://img.shields.io/badge/-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/code/colewelkins/ves3-0?scriptVersionId=128594480)
[![Visit Scroll Prize](https://img.shields.io/badge/Visit-Scroll%20Prize-green)](https://scrollprize.org/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![Framework](https://img.shields.io/badge/Framework-Keras-orange.svg)](https://keras.io/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Vesuvius_Challenge-green.svg)](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/)
[![Paper](https://img.shields.io/badge/Paper-Read-blue.svg)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215775)

## Overview

The goal of this project is to develop a deep learning solution for the detection of ink in 3D x-ray scans. The presence of ink in these scans is of particular interest in various applications, such as the analysis of historical manuscripts, artwork, and archaeological artifacts. Accurate ink detection can provide valuable insights into the composition and preservation of these objects.

To achieve this goal, a Convolutional Neural Network (CNN) model is employed, which is a type of deep learning architecture well-suited for image analysis tasks. CNNs have the ability to automatically learn hierarchical features from raw image data, making them a powerful tool for image classification and segmentation tasks.

In this project, the CNN model is specifically designed to process 3D x-ray scans. The input to the model is a subvolume extracted from the 3D x-ray scan, where each subvolume represents a small region of the scan. The model processes the subvolume and outputs a binary prediction indicating whether ink is present in the subvolume.

## Model Architecture and Data Augmentation

The architecture of the CNN model is designed to handle 3D image data. The model consists of several convolutional layers, each followed by batch normalization and a ReLU activation function. Max pooling is applied after certain convolutional layers to reduce spatial dimensions. The output of the convolutional layers is then flattened and passed through fully connected layers with dropout regularization. The final layer uses a sigmoid activation function to produce the binary prediction.

Data augmentation techniques, including random horizontal and vertical flips, are introduced to the training pipeline to improve the model's robustness and generalization capabilities.

The architecture of the model can be mathematically represented as follows:

Let $X$ be the input subvolume of shape $(C, Z, H, W)$, where $C$ is the number of channels, $Z$ is the depth, $H$ is the height, and $W$ is the width.

The forward pass of the model is given by:

$$
\begin{align*}
X_1 &= \text{ReLU}(\text{BN}(\text{Conv3D}(X, W_1))) \\
X_2 &= \text{MaxPool}(\text{ReLU}(\text{BN}(\text{Conv3D}(X_1, W_2)))) \\
X_3 &= \text{MaxPool}(\text{ReLU}(\text{BN}(\text{Conv3D}(X_2, W_3)))) \\
X_4 &= \text{Flatten}(X_3) \\
X_5 &= \text{ReLU}(\text{FC}(X_4, W_4)) \\
X_6 &= \text{Dropout}(X_5) \\
Y &= \text{Sigmoid}(\text{FC}(X_6, W_5))
\end{align*}
$$

Where:
- $\text{Conv3D}$: 3D Convolution operation
- $\text{BN}$: Batch Normalization
- $\text{ReLU}$: Rectified Linear Unit activation function
- $\text{MaxPool}$: Max Pooling operation
- $\text{FC}$: Fully Connected layer
- $\text{Dropout}$: Dropout regularization
- $\text{Sigmoid}$: Sigmoid activation function
- $W_i$: Learnable weights for each layer

![Model Architecture](https://scrollprize.org/img/tutorials/ml-overview-alpha.png)  
[_source_](https://scrollprize.org/tutorial/4-ink-detection)

## Training Process

The training process for ink detection involves several key steps, as illustrated in the diagram above. Here, a detailed explanation of each step is provided:

- (a) Starting with a fragment of the papyrus.
- (b) Obtaining a 3D volume from the fragment using X-ray imaging techniques.
- (c) Segmenting a mesh from the 3D volume. This mesh represents the surface of the papyrus fragment.
- (d) Sampling a surface volume around the mesh. This surface volume contains the intensity values around the papyrus surface and serves as the input data for the model.
- (e) Taking an infrared photo of the fragment. The infrared photo allows for the visualization of the ink on the papyrus surface.
- (f) Aligning the infrared photo with the surface volume. This alignment ensures that the ink regions in the photo correspond to the appropriate regions in the surface volume.
- (g) Manually creating a binary label image from the infrared photo. In this label image, inked regions are represented as white pixels, while non-inked regions are represented as black pixels.

Once the surface volume and binary label image are prepared, the training of the model proceeds as follows:

- A pixel is selected in the binary label image. The value of this pixel indicates whether there is ink (1) or no ink (0) at that location.
- A subvolume is sampled around the same coordinates from the surface volume. This subvolume serves as the input to the model.
- The subvolume is fed into the model, which produces a prediction of whether ink is present or not.
- The model's prediction is compared with the known label from the binary label image.
- Backpropagation is used to update the model weights based on the difference between the prediction and the ground truth label.

Through this iterative training process, the model learns to recognize patterns associated with the presence of ink in the 3D surface volumes. The trained model can then be used to detect ink in new, unseen data.


## Evaluation

The evaluation of the model is performed on a subset of the dataset that was not used during training. The model processes subvolumes from the evaluation subset and generates binary predictions for the presence of ink. These predictions are compared to the ground truth labels to assess the model's accuracy.

To visualize the results, binary masks representing the predicted ink regions are generated and compared to the ground truth masks. Additionally, quantitative metrics such as precision, recall, and F1-score are calculated to evaluate the model's performance.

![Evaluation Results](https://scrollprize.org/img/tutorials/ink-detection-anim3-dark.jpg)  
[_source_](https://scrollprize.org/tutorial/4-ink-detection)

_The animation above shows the evaluation results. The left side displays the binary mask predicted by the model, while the right side shows the ground truth mask. The model's predictions closely match the ground truth._

## Generating Output

To generate the final output for submission, the trained model is applied to the entire 3D x-ray scan and binary predictions are generated for each subvolume. The binary predictions are then combined to create a binary mask representing the predicted ink regions in the entire scan.

To encode the binary mask for submission, run-length encoding (RLE) is used, which is a lossless compression method that represents consecutive regions of the same value with a single pair of values: the starting position and the length of the region.

The RLE-encoded binary mask is saved to a CSV file, which can be submitted for evaluation.

## Conclusion

This project demonstrates the application of deep learning techniques for the detection of ink in 3D x-ray scans. The trained CNN model is capable of accurately identifying ink regions in the scans, providing valuable information for the analysis and preservation of historical and archaeological objects.

The model can be further improved by experimenting with different architectures, hyperparameters, and data augmentation techniques. Additionally, the model can be extended to handle multi-class segmentation tasks, where different types of ink or materials need to be identified.

Overall, this project showcases the potential of deep learning in the field of image analysis and its ability to provide meaningful insights from complex data.

_The electron microscope images below show the difference between inked and non-inked regions of a manuscript. The top images (A and B) show the surface of the manuscript, while the bottom image (C) shows a side view. The presence of ink can be observed as raised regions on the surface._

![Electron Microscope Images](https://scrollprize.org/img/tutorials/sem-alpha.png)  
[_source_](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215775)

_Electron microscope pictures from the top (A and B) and the side (C)_

## Challenges

The main challenges for ink detection are:
- Model performance: Improving the legibility of detected letters.
- Applying the models to the full scrolls.
- Reverse engineering the models to better understand the patterns used to detect ink.
- Creating more ground truth data (e.g., "campfire scrolls").

## References

- [Vesuvius Challenge: Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/)
- [Ink Detection Tutorial on Kaggle](https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial)
- [From invisibility to readability: Recovering the ink of Herculaneum](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0215775&type=printable)
