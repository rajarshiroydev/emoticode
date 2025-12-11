# models.py
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
import config

class LaBSEGemmaClassifier(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, num_labels=config.NUM_LABELS, device=config.DEVICE):
        super().__init__()
        self.device = device
        
        print(f"Loading Backbone: {config.LLM_MODEL} (Native Precision)...")
        
        # FIX: Removed BitsAndBytesConfig (Quantization causes SegFaults on new archs)
        # Gemma 1B is small enough (2GB VRAM) to run in full Float16/BFloat16.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.gemma_backbone = AutoModel.from_pretrained(
            config.LLM_MODEL,
            dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation="eager" # Stability fix for new models
        )
        
        # Note: prepare_model_for_kbit_training is NOT needed for standard LoRA
        
        # 3. LoRA Adapters
        # We target specific linear layers available in Gemma
        peft_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type=TaskType.FEATURE_EXTRACTION 
        )
        self.gemma_backbone = get_peft_model(self.gemma_backbone, peft_config)
        self.gemma_backbone.print_trainable_parameters()
        
        # 4. Projector (LaBSE -> Gemma)
        self.hidden_size = self.gemma_backbone.config.hidden_size 
        
        # We cast the projector to match the backbone dtype
        self.projector = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(device=device, dtype=dtype)
        
        # 5. Classification Head
        self.classifier = nn.Linear(self.hidden_size, num_labels).to(device=device, dtype=dtype)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels=None):
        # embeddings: [Batch, 768]
        # Ensure input is the correct dtype
        embeddings = embeddings.to(dtype=self.projector[0].weight.dtype)
        
        projected = self.projector(embeddings)     # -> [Batch, Gemma_Dim]
        inputs_embeds = projected.unsqueeze(1)     # -> [Batch, 1, Gemma_Dim]
        
        # Pass through LLM
        outputs = self.gemma_backbone(inputs_embeds=inputs_embeds)
        
        # Pool (First Token)
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
            
        return logits, loss