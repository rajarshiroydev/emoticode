# train.py
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report, f1_score
import tqdm
import config
import sys
import os

# --- 1. Define the Prompt-Augmented Classifier ---
class PromptGemmaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        print(f"Loading Gemma-3-1B (Float32 Mode)...")
        # Load Model
        self.gemma = AutoModel.from_pretrained(
            "google/gemma-3-1b-it",
            device_map={"": config.DEVICE},
            trust_remote_code=True,
            attn_implementation="eager" 
        )
        
        # Load Tokenizer (Required for Prompt Priming)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        
        # Create Static Prompt Inputs (Frozen)
        prompt_text = "Analyze the emotion of this sentence:"
        self.prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(config.DEVICE)
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=16, lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=0.05, bias="none", 
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.gemma = get_peft_model(self.gemma, peft_config)
        
        hidden_size = self.gemma.config.hidden_size
        
        # Projector: Maps NLLB(1024) -> Gemma(2048)
        self.projector = nn.Sequential(
            nn.Linear(config.INPUT_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(device=config.DEVICE, dtype=torch.float32)
        
        # Classifier Head
        self.classifier = nn.Linear(hidden_size, config.NUM_LABELS).to(device=config.DEVICE, dtype=torch.float32)
        
        # Loss with Label Smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, vectors, labels=None):
        batch_size = vectors.shape[0]
        
        # 1. Project Input Vector
        # [Batch, 1024] -> [Batch, 2048]
        proj_embeds = self.projector(vectors.to(dtype=torch.float32))
        # Reshape to [Batch, 1, 2048]
        proj_embeds = proj_embeds.unsqueeze(1)
        
        # 2. Get Prompt Embeddings (THE FIX)
        # We use .get_input_embeddings() which is safer than accessing .model.embed_tokens
        embedding_layer = self.gemma.get_base_model().get_input_embeddings()
        prompt_embeds = embedding_layer(self.prompt_tokens)
        
        # Expand prompt to match batch size
        # [Batch, Seq_Len, 2048]
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
        
        # 3. Concatenate: [Prompt] + [Sentence Vector]
        inputs_embeds = torch.cat([prompt_embeds, proj_embeds], dim=1)
        
        # 4. Pass through Gemma
        outputs = self.gemma(inputs_embeds=inputs_embeds)
        
        # 5. Pool the LAST token (The Sentence Vector)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        
        # 6. Classify
        logits = self.classifier(last_hidden)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss

# --- 2. Main Training Loop ---
def main():
    print("=== STEP 2: Training Gemma (Prompt Augmented) ===")
    
    # Check paths
    if not os.path.exists(config.TRAIN_EMBEDS):
        print("❌ Error: Embeddings not found. Run embed.py first.")
        return

    # Load Data
    X_train = torch.load(config.TRAIN_EMBEDS)
    X_val = torch.load(config.VAL_EMBEDS)
    X_test = torch.load(config.TEST_EMBEDS)
    
    df_train = pd.read_csv(config.TRAIN_CSV)
    df_val = pd.read_csv(config.VAL_CSV)
    df_test = pd.read_csv(config.TEST_CSV)

    # Filter labels
    df_train = df_train[df_train['language'].isin(config.LANG_MAP.keys())]
    df_val = df_val[df_val['language'].isin(config.LANG_MAP.keys())]
    
    y_train = torch.tensor(df_train['emotion'].map(config.EMOTION_MAP).values, dtype=torch.long)
    y_val = torch.tensor(df_val['emotion'].map(config.EMOTION_MAP).values, dtype=torch.long)
    
    # Loaders
    BATCH_SIZE = config.BATCH
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # Init Model
    model = PromptGemmaClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training Config
    EPOCHS = config.EPOCH
    best_macro_f1 = 0.0
    BEST_MODEL_PATH = "best_model_state.pth"
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for bx, by in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            bx, by = bx.to(config.DEVICE), by.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits, loss = model(bx, labels=by)
            loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(config.DEVICE)
                logits, _ = model(bx)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.numpy())
        
        avg_loss = train_loss / len(train_loader)
        macro_f1 = f1_score(val_targets, val_preds, average='macro')
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🌟 New Best Model! Saved to {BEST_MODEL_PATH}")

    # ==========================================
    # REPORT
    # ==========================================
    print("\n" + "="*40)
    print(f"✅ Training Complete. Best Macro F1: {best_macro_f1:.4f}")
    print("="*40)
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    final_preds = []
    final_targets = []
    
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(config.DEVICE)
            logits, _ = model(bx)
            preds = torch.argmax(logits, dim=1)
            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(by.numpy())
            
    target_names = list(config.EMOTION_MAP.keys())
    print("\n--- Validation Set Classification Report ---")
    print(classification_report(final_targets, final_preds, target_names=target_names, digits=4))
    
    # ==========================================
    # SUBMISSION
    # ==========================================
    print("\n--- Generating Submission File ---")
    test_loader = DataLoader(TensorDataset(X_test), batch_size=BATCH_SIZE)
    test_preds = []
    
    with torch.no_grad():
        for bx in tqdm.tqdm(test_loader, desc="Processing Test"):
            bx = bx[0].to(config.DEVICE)
            logits, _ = model(bx)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            
    sub = pd.DataFrame({
        'id': df_test['id'],
        'emotion': [config.ID_TO_EMOTION[p] for p in test_preds]
    })
    sub.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"✅ Submission saved to {config.SUBMISSION_FILE}")

if __name__ == "__main__":
    main()