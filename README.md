# deep-learning-image-classification-benchmarks

🧠 Repository Description (README Intro)

A comparative deep learning project exploring multiple convolutional neural network architectures for image classification on the CIFAR-10 dataset.

The project implements and evaluates three approaches: a custom CNN, a ResNet architecture built from scratch, and a transfer learning pipeline using pretrained ResNet18, highlighting performance trade-offs and architectural design choices.

🚀 Key Features
🧱 Custom CNN Architecture
Multi-layer convolutional network with:
Batch normalization
Dropout regularization
Learning rate scheduling for stable training
→ see:
🧠 ResNet Built From Scratch
Implementation of residual blocks (skip connections)
Deep architecture with:
Layer stacking
Identity mapping
Demonstrates understanding of modern deep learning architectures
→ see:
⚡ Transfer Learning (ResNet18)
Pretrained ImageNet weights
Selective fine-tuning:
Frozen backbone
Trainable final layers
Efficient training with reduced compute
→ see:
📊 Model Evaluation
Test accuracy measurement
Confusion matrix visualization
Comparative analysis across architectures
🔁 Training Optimization Techniques
Learning rate scheduling
Data augmentation:
Random crop
Horizontal flip
GPU acceleration support
⚙️ Tech Stack
Python
PyTorch
Torchvision
Matplotlib
Scikit-learn
