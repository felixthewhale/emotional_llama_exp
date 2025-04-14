# Emotional Llama (Simplified)

This project is an experimental implementation testing an architecture for emotional modulation in Llama; while it achieves some modulation, it primarily serves as a structural exploration.
This repository contains an implementation exploring the integration of emotion modulation into a Llama-based large language model. The goal is to enable the model to adjust its behavior based on specified emotional context, provided through an "emotion vector".

## Overview

The core modification involves augmenting a standard Hugging Face Llama model (specifically `meta-llama/Llama-3.2-3B` by default) to accept an additional input: an emotion vector. This vector represents emotional states along several dimensions.

Currently, the implementation uses **8 emotion dimensions**, defined as continuous bipolar axes:

*   SADNESS_JOY
*   FEAR_COURAGE
*   DISGUST_ACCEPTANCE
*   ANGER_CALMNESS
*   SURPRISE_EXPECTATION
*   DISTRUST_TRUST
*   BOREDOM_INTEREST
*   INDIFFERENCE_EMPATHY

The model processes this emotion vector using a dedicated projection layer (`emotion_proj_embed`) and incorporates the resulting emotional representation into the model's input embeddings. This allows the emotional context to influence the subsequent token generation process.

## Key Components

*   **`EmotionalLlamaModel`**: A custom model class inheriting from `transformers.LlamaForCausalLM`. It overrides the `forward` method to handle the `emotion_vector` input.
*   **`emotion_proj_embed`**: A small neural network (currently a Multi-Layer Perceptron) within `EmotionalLlamaModel`. It projects the input `emotion_vector` (dimension `EMOTION_DIMENSIONS`) to the model's hidden dimension size. This projected vector is then added element-wise to the token embeddings.
*   **`DataCollatorForEmotionalLlama`**: A custom data collator for use with the Hugging Face `Trainer`. It ensures that `emotion_vector` data is correctly batched and padded alongside `input_ids` and `attention_mask` during training.
*   **`train_emotional_llama`**: The main script function that orchestrates the loading of the model and tokenizer, dataset preparation (requires a `dataset.json` and associated helper functions), optional PEFT/LoRA setup, custom optimizer configuration, and initiating the training process via the `Trainer`.
*   **`CustomTrainer` & Optimizer Setup**: A slight modification to the `Trainer` and a specific optimizer configuration are used. This addresses the observation that the `emotion_proj_embed` layer often requires a significantly different (usually higher) learning rate compared to the base model parameters (or LoRA adapters) for effective training. The custom optimizer applies different learning rates and weight decay settings to these parameter groups.

## Configuration

Primary configuration options are defined as constants at the beginning of `emotional_llama_simplified.py`:

*   **`MODEL_NAME`**: Specifies the base Hugging Face model identifier (default: `"meta-llama/Llama-3.2-3B"`).
*   **`EMOTION_DIMENSIONS`**: Defines the size of the emotion vector (default: `8`). Ensure your dataset's emotion vectors match this dimension.

## Usage

1.  **Dataset Preparation**: Prepare your training data in a JSON format (e.g., `dataset.json`). Each data entry should contain the text sequence and corresponding `emotion_vectors`. A helper function (like `create_huggingface_dataset`, assumed to be in `dataset.py`) is needed to load this JSON and format it into a Hugging Face `Dataset` object containing `input_ids`, `attention_mask`, and `emotion_vectors`. The `emotion_vectors` should be structured appropriately (e.g., a list of vectors, one per token, or a single vector per sequence, handled by the collator).
2.  **Dependencies**: Install necessary libraries: `torch`, `transformers`, `peft`, `datasets`, `accelerate`.
3.  **Configuration**: Modify the `dataset_path` and `output_dir` variables within the `if __name__ == "__main__":` block in `emotional_llama_simplified.py` to point to your dataset file and desired model output location.
4.  **Execution**: Run the training script using `python emotional_llama_simplified.py`.
5.  **Hyperparameter Tuning**: Adjust training parameters (learning rates for LoRA and emotion projection, epochs, batch size, LoRA configuration, etc.) within the `train_emotional_llama` function as needed.

## Implementation Notes & Potential Future Work

*   **Emotion Injection Point**: The current approach injects the emotional information at the embedding layer ("early modulation"). Exploring alternative injection points (e.g., modulating attention layers, later hidden states) could be a direction for future work.
*   **LoRA Usage**: The script defaults to using PEFT/LoRA (`use_lora=True`) for parameter-efficient fine-tuning. Only the LoRA adapters and the `emotion_proj_embed` layer are trained.
*   **Learning Rate Differential**: As noted, the emotion projection layer may benefit from a distinct, often higher, learning rate than other trainable parameters. The custom optimizer configuration reflects this experimental finding.
*   **Debugging Output**: The code includes several `print` statements, particularly within the `forward` pass and data loading loop, for debugging purposes. These may be removed for cleaner execution.
*   **Dataset Dependency**: The effectiveness of the model heavily relies on the quality and format of the training data, specifically how emotion vectors are provided and aligned with the text sequences.
*   **Emotion Source**: This implementation assumes that emotion vectors are provided as input during training and inference. It does not include mechanisms for dynamically inferring emotion from the text context itself.

## Author

FelixTheWhale
