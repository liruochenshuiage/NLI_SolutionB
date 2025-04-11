---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/liruochenshuiage/NLI_SolutionB

---

# Model Card for q36172hw-m95082rl-NLI-SolutionB


This is a classification model that was trained to
      determine the logical relationship between two given sentences (natural language inference task).


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is a Bidirectional LSTM (BiLSTM) architecture trained from scratch 
    on a natural language inference dataset. It takes paired sentences (premise and hypothesis) as input 
    and classifies their relationship as entailment, contradiction, or neutral.

- **Developed by:**  Ruochen Li and Haotian Wu
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** BiLSTM
- **Finetuned from model [optional]:** BiLSTM

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://keras.io/api/layers/recurrent_layers/lstm/
- **Paper or documentation:** N/A

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24,432 pairs of sentences from a natural language inference (NLI) dataset. Each pair consists of a premise and a hypothesis, with labels indicating their logical relationship (entailment, contradiction, or neutral).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 5e-4
      - train_batch_size: 64
      - num_epochs: 10
      - embedding_dim: 200
      - lstm_units: 128
      - dropout_rate: 0.3
      - max_sequence_length: 70
    

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: ～30 minutes
      - duration per training epoch: 3 minutes
      - model size: 18.5MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 2K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 67.5% and an accuracy of 67.4%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: P100

### Software


      - Python 3.9+
      - TensorFlow 2.14.0
      - scikit-learn 1.3.0
      - NumPy 1.23.0
    

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Since the model requires fixed-length input sequences, 
    any input that exceeds the maximum length (set based on the 95th percentile of the training data) 
    will be truncated. This may lead to partial information loss, especially for unusually long inputs.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The model hyperparameters were selected through grid search 
      over a range of embedding dimensions, LSTM units, and dropout rates. 
      Each combination was evaluated based on weighted F1 score on the development set. 
      The best-performing model used an embedding dimension of 200, 128 LSTM units, and a dropout rate of 0.3.

Model URL：https://drive.google.com/file/d/177nzy7mqlgnjNIy7yFzi1JcmTpph8fLe/view?usp=sharing
