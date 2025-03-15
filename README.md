# Financial-Sentiment-Distillation

## Overview

**Financial-Sentiment-Distillation** is a sentiment analysis project focused on financial news, utilizing a teacher-student knowledge distillation approach. The goal is to pre-train a compact and efficient language model specialized in financial sentiment analysis, leveraging domain-specific knowledge from a larger pre-trained model.

## Features

- **Knowledge Distillation:** Implements a teacher-student framework where a smaller student model learns from a larger, pre-trained teacher model.
- **Domain-Specific Fine-Tuning:** Focuses on financial news to enhance the model's understanding of domain-specific language.
- **Data Augmentation:** Employs techniques like synonym replacement to enrich the dataset and improve model generalization.
- **Evaluation Metrics:** Includes detailed performance evaluation using accuracy, classification reports, and confusion matrices.
- **Visualization:** Provides loss and accuracy plots to monitor training and validation performance.

## Dataset

The dataset used is a financial news dataset from Kaggle, consisting of labeled news articles categorized into positive, negative, and neutral sentiments.

## Model Architecture

- **Teacher Model:** `finbert` (a BERT-based model fine-tuned on financial texts).
- **Student Model:** `distilbert-base-uncased` further fine-tuned using knowledge distillation.

The student model architecture includes:
- DistilBERT encoder
- Layer normalization
- Global average pooling
- Fully connected layers with dropout
- Softmax output layer for sentiment classification

## Knowledge Distillation

The distillation process uses Kullback-Leibler Divergence (KL Divergence) to align the student model's output with the teacher model, combined with traditional cross-entropy loss to maintain accuracy on labeled data.

## Data Augmentation

Synonym replacement is applied to the training data to introduce variability and improve model robustness. This technique helps the model generalize better to unseen data.

## Training

- Optimizer: Adam with a learning rate of 5e-5
- Loss Function: Custom distillation loss combining KL Divergence and cross-entropy
- Batch Size: 32
- Epochs: 20 (with early stopping and learning rate reduction callbacks)

## Evaluation

The model achieved an accuracy of approximately 76% on the validation set. Evaluation includes:

- Accuracy and loss plots
- Confusion matrix
- Classification report detailing precision, recall, and F1-scores

## Visualization

- Training and validation loss/accuracy curves
- Confusion matrix heatmap for visualizing model predictions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vedavyas0105/FinSentDistill.git
   ```

2. Navigate to the project directory:
   ```bash
   cd FinSentDistill
   ```

## Usage

1. Prepare the dataset and place it in the project directory.
2. Run the training script:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```
4. Visualize results:
   ```bash
   python visualize.py
   ```

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for the pre-trained teacher model
- Kaggle for the financial news dataset
- Hugging Face Transformers library for model implementations

