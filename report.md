# Collaborative CNN Cross-Dataset Evaluation Report  
## Team 12– Tejas(2025CSZ0008) & Divya(2025AIZ0019)
# 1. Introduction

This project investigates the cross-dataset generalization of two independently trained convolutional neural network (CNN) models. Each team member trained a model on their respective dataset, and we subsequently tested each model on the other member's dataset to evaluate robustness.

The main objectives are:

To analyze how CNN models trained on different datasets perform on unseen data.

To compare the generalization performance of Model V1 and Model V2.

To demonstrate proper GitHub collaboration workflow, including:

Forking and cloning repositories

Branching strategies

Pull requests and code merging

Issue tracking

Both models use ResNet-18 with transfer learning, but are trained on different datasets, with slight variations in augmentation and fine-tuning strategies.

# 2. Dataset 

## **Dataset 1**
https://www.kaggle.com/datasets/tongpython/cat-and-dog

## **Dataset 2**
https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition

### **Key Observation**
Both datasets have the **same number of classes**, but:
- class distribution  
- background variations  
- image quality  
- camera sources  
- illumination  


# 3. Model Architectures
Model V1

Backbone: ResNet-18 pretrained on ImageNet

Modification: Final fully connected layer replaced for binary classification

Training Characteristics:

Mild data augmentations

Optimizer: Adam

Loss: CrossEntropyLoss

Model V2

Backbone: ResNet-18 pretrained on ImageNet

Modification: Final fully connected layer replaced for binary classification

Training Characteristics:

Slightly different augmentation strategy

Fine-tuning dynamics vary

Trained on a distinct dataset

Both models rely on transfer learning, but subtle differences in dataset diversity and training approach may impact generalization
# 4. Experimental Results

4.1 Self-Evaluation Metrics

Model V2 on dataset_2

Metric	       Score

Accuracy	 = 0.9814

Precision	=  0.9816

Recall	  =  0.9814

F1-Score	 = 0.9814


4.2 Cross-Testing Results

Model V1 tested  dataset_2

Metric	      Score

Accuracy	=  0.9810

Precision = 	0.9813

Recall	  =  0.9810

F1-Score	 = 0.9810



# 5. Observations & Analysis

Dataset Impact
Minor variations in performance are likely influenced by dataset distribution, image diversity, and augmentation strategies.

Transfer Learning Effectiveness
Both models leveraged ResNet-18 effectively:

Model V2 provides a stable and high-performing baseline.

Model V1 demonstrates comparable robustness, potentially due to slightly stronger augmentation or training strategies.

Cross-Domain Generalization

Both models retain most of their performance on unseen datasets, indicating that the learned features are generalizable.

Slight differences highlight the influence of training data diversity and hyperparameter choices.

# 6. Conclusion
This collaborative project highlights several key points:

Two similar CNN architectures can exhibit minor differences in performance due to dataset and training variations.

Dataset diversity and augmentation play a crucial role in model generalization.

Cross-dataset testing provides a valuable measure of robustness beyond traditional self-evaluation.

Collaborative GitHub workflows allow for efficient team-based model development and evaluation.








