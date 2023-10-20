# Fine-tuning Hyperparameters

Below is a description of the hyperparameters used in the `ai_worker/fine_tune.py` script, with their default values and purposes:

## 1. `lora_alpha`
- **Default Value:** 64
- **Description:** 
  This parameter refers to the value of `alpha` in the LORA (Low-Rank Adaptation) setup. It can control the amount of rank adaptation when tuning the model.

## 2. `lora_dropout`
- **Default Value:** 0.05 (5%)
- **Description:** 
  The dropout rate associated with the LORA layer. Dropout is a regularization technique to prevent overfitting in neural networks. A higher value might increase regularization but could reduce the learning capacity of the model.

## 3. `batch_size`
- **Default Value:** 4
- **Description:** 
  The number of samples processed before the model's internal parameters are updated. A larger batch size can increase the training speed but might require more memory.

## 4. `accumulation_steps`
- **Default Value:** 4
- **Description:** 
  Gradient accumulation steps are used to simulate a larger batch size when memory limitations prevent using a larger actual batch size. For instance, if you want an effective batch size of 16 but can only fit 4 samples in memory at a time, you'd use a `batch_size` of 4 and `accumulation_steps` of 4. The gradients from each batch are accumulated over the set number of steps before being applied.

## 5. `n_steps`
- **Default Value:** 500
- **Description:** 
  The maximum number of training steps. It limits the total number of iterations the training process will go through, regardless of the number of epochs.

## 6. `n_epochs`
- **Default Value:** 500
- **Description:** 
  The number of complete passes through the training dataset. One epoch means that each sample in the training dataset has been used once to update the model's weights.

## 7. `learning_rate_multiplier`
- **Default Value:** 2.5e-5
- **Description:** 
  This hyperparameter multiplies the base learning rate. The learning rate determines the size of the steps taken to adjust the model's weights during training. A smaller value will lead to slower convergence but might achieve better results, while a larger value could speed up convergence but risk overshooting the optimal weights.

## 8. `training_split`
- **Default Value:** 0.8 (80%)
- **Description:** 
  This determines the fraction of the dataset that should be used for training. The remaining portion (1 - `training_split`) is typically used for validation. For instance, a value of 0.8 means 80% of the data is used for training and 20% for validation.

## 8. `stop_eval_loss`
- **Default Value:** 0.01
- **Description:** 
  Stops training when eval loss drops below this number.

When using `ai_worker/fine_tune.py`, you can customize these hyperparameters as needed, bearing in mind their potential effects on training performance and model generalization.
