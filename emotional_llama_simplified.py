# emotional_llama_simplified.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader

# Constants
MODEL_NAME = "meta-llama/Llama-3.2-3B"
EMOTION_DIMENSIONS = 8
EMOTION_DIMENSIONS_REFERENCE = [
    "SADNESS_JOY", "FEAR_COURAGE", "DISGUST_ACCEPTANCE", "ANGER_CALMNESS",
    "SURPRISE_EXPECTATION", "DISTRUST_TRUST", "BOREDOM_INTEREST", "INDIFFERENCE_EMPATHY"
]

class EmotionalLlamaModel(LlamaForCausalLM):
    """Emotional LLaMA model with emotion modulation."""
    def __init__(self, config):
        super().__init__(config)
        self.emotion_dim = EMOTION_DIMENSIONS

        # Test A2 - Early apply, MLP
        self.emotion_proj_embed = nn.Sequential(
            nn.Linear(self.emotion_dim, config.hidden_size // 4), 
            nn.GELU(), 
            nn.Linear(config.hidden_size // 4, config.hidden_size) 
        )

        # Initialization for the MLP
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.emotion_proj_embed.apply(init_weights)
        # # TEST A2 - Early apply, MLP END

        # # TEST A1 - Early apply
        # self.emotion_proj_embed = nn.Linear(self.emotion_dim, config.hidden_size)
        # # torch.nn.init.zeros_(self.emotion_proj_embed.weight)
        # # torch.nn.init.zeros_(self.emotion_proj_embed.bias)
        # # Use Xavier Uniform Initialization for weights
        # torch.nn.init.xavier_uniform_(self.emotion_proj_embed.weight)
        # with torch.no_grad(): # important!
        #     self.emotion_proj_embed.weight.data *= 2
        # if self.emotion_proj_embed.bias is not None:
        #     torch.nn.init.zeros_(self.emotion_proj_embed.bias)
        # # TEST A1 END

        # # Original
        # self.emotion_proj = nn.Sequential(
        #     nn.Linear(self.emotion_dim, config.hidden_size // 2),
        #     # nn.ReLU(),
        #     nn.Linear(config.hidden_size // 2, config.hidden_size),
        #     # nn.Sigmoid()
        # )

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight) # Or "Kaiming"
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
        # self.emotion_proj.apply(init_weights)



        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        emotion_vector: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:

        # 1. Prepare Input Embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify input_ids OR inputs_embeds, not both.")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            # Handle generation case where only past_key_values might exist initially
            # This part might need refinement depending on how HF handles KV cache only calls
            if 'past_key_values' not in kwargs or kwargs['past_key_values'] is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            # Infer from past_key_values if possible (complex, rely on standard handling for now)
            # For the first generation call, inputs_embeds will be derived from input_ids normally.
            # For subsequent calls, only the *new* token's embed is needed usually.
            pass # standard generate handle embedding lookup for subsequent tokens


        # 2. Apply Emotion Modulation to Embeddings
        if emotion_vector is not None and inputs_embeds is not None: # Check inputs_embeds exists
            if emotion_vector.shape[0] != batch_size:
                raise ValueError("Batch size mismatch between emotion_vector and input.")

            # Project emotion vector to hidden size
            # Expected emotion_vector shape: [batch, seq_len, emotion_dim] (from collator)
            # Or [batch, emotion_dim] during simple inference -> need unsqueeze
            if emotion_vector.dim() == 2:
                emotion_vector = emotion_vector.unsqueeze(1) # -> [batch, 1, emotion_dim]

            # Ensure emotion_vector seq_len matches inputs_embeds seq_len
            # This handles training (full seq) and generation (often seq_len=1)
            current_seq_len = inputs_embeds.shape[1]
            if emotion_vector.shape[1] == 1 and current_seq_len > 1:
                # Expand single inference vector to match sequence
                emotion_vector = emotion_vector.expand(-1, current_seq_len, -1)
            elif emotion_vector.shape[1] != current_seq_len:
                # Take the slice relevant to the current input embeddings
                # This handles cases where emotion_vector might be longer than current input segment
                emotion_vector = emotion_vector[:, :current_seq_len, :]


            # Project to hidden size
            emotion_offset = self.emotion_proj_embed(emotion_vector) # -> [batch, current_seq_len, hidden_size]
            
            
            # === START DEBUGGING PRINTS ===
            print(f"\n--- Emotion Offset Debug (Emotion: {'PROVIDED' if emotion_vector is not None else 'NONE'}) ---") # Indicate if emotion was provided
            print(f"  inputs_embeds shape: {inputs_embeds.shape}")
            print(f"  emotion_offset shape: {emotion_offset.shape}")
            print(f"  emotion_offset Stats: min={emotion_offset.min().item():.4f}, max={emotion_offset.max().item():.4f}, mean={emotion_offset.mean().item():.4f}, std={emotion_offset.std().item():.4f}")
            # Stats before adding offset
            embeds_before_mean = inputs_embeds.mean().item()
            embeds_before_std = inputs_embeds.std().item()
            print(f"  Embeddings BEFORE add: mean={embeds_before_mean:.4f}, std={embeds_before_std:.4f}")

            # Add to embeddings
            inputs_embeds = inputs_embeds + emotion_offset # * 0.1

            # === START DEBUGGING PRINTS ===
            # Stats after adding offset
            embeds_after_mean = inputs_embeds.mean().item()
            embeds_after_std = inputs_embeds.std().item()
            print(f"  Embeddings AFTER add:  mean={embeds_after_mean:.4f}, std={embeds_after_std:.4f}")
            print(f"--- End Emotion Offset Debug ---\n")
            # === END DEBUGGING PRINTS ===
        # 3. Pass embeddings (potentially modified) to the base model
        # Crucially, pass inputs_embeds, not input_ids if modified
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # Still need last hidden state
            return_dict=True,
            **kwargs
        )

        # 4. Get final hidden state and compute logits 
        hidden_states = outputs.hidden_states[-1]

        # late-stage modulation may be heree

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DataCollatorForEmotionalLlama:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, examples: list) -> Dict[str, torch.Tensor]:
        input_ids = [example.get("input_ids", []) for example in examples]
        attention_masks = [example.get("attention_mask", []) for example in examples]
        emotion_vectors = [example.get("emotion_vectors", []) for example in examples]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        padded_emotion_vectors = []
        for emotion_vec in emotion_vectors:
            if not emotion_vec:
                padded_vec = [[0.0] * EMOTION_DIMENSIONS] * self.max_length
            else:
                seq_len = len(emotion_vec)
                padding_len = self.max_length - seq_len
                if padding_len > 0:
                    padded_vec = emotion_vec + [[0.0] * EMOTION_DIMENSIONS] * padding_len
                else:
                    padded_vec = emotion_vec[:self.max_length]
            padded_emotion_vectors.append(padded_vec)

        batch["emotion_vector"] = torch.tensor(padded_emotion_vectors, dtype=torch.float)
        batch["labels"] = batch["input_ids"].clone()
        return batch


class CustomTrainer(Trainer):  # Subclass Trainer
    def get_train_dataloader(self) -> DataLoader:
        """
        Override the dataloader creation to use our data collator directly,
        bypassing any potential internal modifications by the Trainer.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Create a DataLoader directly, using data_collator
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,  # for training!
            collate_fn=data_collator, 
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def train_emotional_llama(
    model_name=MODEL_NAME,
    dataset_path="./dataset.json",
    output_dir="./emotional-llama",
    max_length=256,
    learning_rate=1e-5,
    num_train_epochs=20,
    per_device_batch_size=1,
    gradient_accumulation_steps=8,
    use_lora=True
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = EmotionalLlamaModel.from_pretrained(model_name)

    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=32,
            # lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Enable emotion training
    for param in model.emotion_proj_embed.parameters():
        param.requires_grad = True

    from dataset import create_huggingface_dataset
    dataset = create_huggingface_dataset(dataset_path, tokenizer, max_length)
    data_collator = DataCollatorForEmotionalLlama(tokenizer=tokenizer, max_length=max_length)


    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        lr_scheduler_type="cosine", # Common scheduler
        optim="adamw_torch"
    )

    # Split Learning Rates to Global and Emotion Layer. Emotion needs way higher LR otherwise stuck at mean.
    lora_lr = training_args.learning_rate # e.g., 2e-5
    emotion_proj_lr = 1e-3 # Experiment with this (e.g., 5e-4, 1e-3)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    emotion_params = [p for n, p in model.named_parameters() if "emotion_proj" in n and p.requires_grad]
    # Ensure all trainable are covered..
    trainable_param_names = {n for n, p in model.named_parameters() if p.requires_grad}
    lora_param_names = {n for n, p in model.named_parameters() if "lora_" in n and p.requires_grad}
    emotion_param_names = {n for n, p in model.named_parameters() if "emotion_proj" in n and p.requires_grad}
    other_trainable_params = [p for n, p in model.named_parameters() if n in trainable_param_names and n not in lora_param_names and n not in emotion_param_names]

    # Group other trainable params with LoRA
    main_params = lora_params + other_trainable_params

    optimizer_grouped_parameters = [
        # Apply weight decay to LoRA and potentially other non-emotion..
        {"params": main_params, "lr": lora_lr, "weight_decay": training_args.weight_decay},
        # Apply NO weight decay to the emotion projection layer!!
        {"params": emotion_params, "lr": emotion_proj_lr, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),  
    )
   # --- DEBUGGING: Print statements inside the training loop ---
    for epoch in range(int(training_args.num_train_epochs)): 
        print(f"Starting Epoch {epoch + 1}/{int(training_args.num_train_epochs)}")
        for step, batch in enumerate(trainer.get_train_dataloader()):
            print(f"  Step {step + 1}:")
            # print(f"    input_ids shape: {batch['input_ids'].shape}")
            # print(f"    attention_mask shape: {batch['attention_mask'].shape}")
            # print(f"    emotion_vector shape: {batch['emotion_vector'].shape}")
            # print(f"    labels shape: {batch['labels'].shape}")
            # # Print first 5 for checking.
            # print(f"    input_ids (first 5): {batch['input_ids'][:5]}")
            # print(f"    attention_mask (first 5): {batch['attention_mask'][:5]}")
            # print(f"    emotion_vector (first 5): {batch['emotion_vector'][:5]}")
            # print(f"    labels: (first 5){batch['labels'][:5]}")
            # # 
            print(f"    emotion_vector batch MIN: {batch['emotion_vector'].min()}")
            print(f"    emotion_vector batch MAX: {batch['emotion_vector'].max()}")
            print(f"    emotion_vector batch MEAN: {batch['emotion_vector'].mean()}")
    # Train model
    trainer.train()
    trainer.save_model(output_dir)
    return model, tokenizer

if __name__ == "__main__":
    train_emotional_llama(
        dataset_path="C:\\dataset_emotion.json",
        output_dir="./emotional-llama-output"
    )
