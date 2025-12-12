import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report
import tqdm
import config
import os

# ==========================================
# 1. MODEL DEFINITION (Must Match Training)
# ==========================================
class HybridGemmaClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        print(f"Loading Gemma-3-1B-IT Architecture...")
        # Load Model
        self.gemma = AutoModel.from_pretrained(
            "google/gemma-3-1b-it", 
            trust_remote_code=True,
            attn_implementation="eager" 
        )
        
        # Enable LoRA (Must match saved config)
        peft_config = LoraConfig(
            r=16, lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )
        self.gemma = get_peft_model(self.gemma, peft_config)
        
        # Feature Projector
        self.embed_dim = self.gemma.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        ).to(dtype=torch.float32)
        
        self.classifier = nn.Linear(self.embed_dim, config.NUM_LABELS)

    def forward(self, input_ids, attention_mask, vectors):
        # 1. Get Text Embeddings safely
        embedding_layer = self.gemma.get_input_embeddings()
        text_embeds = embedding_layer(input_ids)
        
        # Scale embeddings manually (Gemma requirement)
        text_embeds = text_embeds * (self.embed_dim ** 0.5)
        
        # 2. Project the Fusion Vector
        vector_embeds = self.projector(vectors.to(dtype=torch.float32)).unsqueeze(1)
        
        # 3. Concatenate
        inputs_embeds = torch.cat([vector_embeds, text_embeds], dim=1)
        
        # 4. Adjust Attention Mask
        ones = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        extended_attention_mask = torch.cat([ones, attention_mask], dim=1)
        
        # 5. Pass to Gemma
        outputs = self.gemma(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
        
        # 6. Classify Last Token
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return self.classifier(last_hidden)

# ==========================================
# 2. DATALOADER (Same as Training)
# ==========================================
def create_dataloader(df, embeddings, labels=None, tokenizer=None, batch_size=32):
    # NOTE: If you used Romanization in training, add the function call here!
    # df = apply_romanization(df) 
    
    print("   -> Tokenizing Text...")
    encodings = tokenizer(
        df['Sentence'].tolist(), 
        padding=True, 
        truncation=True, 
        max_length=64, 
        return_tensors="pt"
    )
    
    ids = encodings['input_ids']
    mask = encodings['attention_mask']
    
    if labels is not None:
        dataset = TensorDataset(ids, mask, embeddings, labels)
    else:
        dataset = TensorDataset(ids, mask, embeddings)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 3. MAIN EVALUATION
# ==========================================
def main():
    MODEL_PATH = "best_hybrid.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        return

    print("=== Generating Classification Report ===")
    
    # Load Data
    print("1. Loading Data & Embeddings...")
    df_val = pd.read_csv(config.VAL_CSV)
    X_val = torch.load(config.VAL_EMBEDS)
    
    # Filter Languages
    df_val = df_val[df_val['language'].isin(config.LANG_MAP.keys())]
    y_val = torch.tensor(df_val['emotion'].map(config.EMOTION_MAP).values, dtype=torch.long)
    
    # Load Tokenizer
    print("2. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    
    # Create Loader
    val_loader = create_dataloader(df_val, X_val, y_val, tokenizer, batch_size=config.BATCH)
    
    # Initialize Model
    print("3. Initializing Model...")
    model = HybridGemmaClassifier(input_dim=config.INPUT_DIM).to(config.DEVICE)
    
    # Load Weights
    print(f"4. Loading Weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE))
    except Exception as e:
        print(f"⚠️ Error loading state_dict directly: {e}")
        print("   Attempting Strict=False loading (in case of LoRA keys mismatch)...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE), strict=False)

    model.eval()
    
    # Inference
    print("5. Running Inference...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for b_ids, b_mask, b_vec, b_lbl in tqdm.tqdm(val_loader):
            b_ids = b_ids.to(config.DEVICE)
            b_mask = b_mask.to(config.DEVICE)
            b_vec = b_vec.to(config.DEVICE)
            
            outputs = model(b_ids, b_mask, b_vec)
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_targets.extend(b_lbl.numpy())

    # Report
    print("\n" + "="*50)
    print("FINAL CLASSIFICATION REPORT")
    print("="*50)
    
    target_names = list(config.EMOTION_MAP.keys())
    
    # Detailed Report
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))
    
    # Save to file just in case
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("classification_report.csv")
    print("📄 Report saved to classification_report.csv")

if __name__ == "__main__":
    main()