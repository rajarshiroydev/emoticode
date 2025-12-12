import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report, f1_score
import tqdm
import config
import os

# --- Mixup Helper Functions ---
def mixup_data(vectors, labels, alpha=0.2):
    '''
    Applies Mixup to VECTORS only. 
    Returns mixed vectors, target_a, target_b, and lambda.
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = vectors.size()[0]
    index = torch.randperm(batch_size).to(config.DEVICE)

    # Mix the continuous vectors
    mixed_vectors = lam * vectors + (1 - lam) * vectors[index, :]
    
    # We maintain the text/labels from the original batch (A), 
    # but we will calculate loss against both A and B.
    y_a, y_b = labels, labels[index]
    return mixed_vectors, y_a, y_b, lam, index

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Classifier Model ---
class HybridGemmaClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        print(f"Loading Gemma-3-1B-IT...")
        # Load Model
        self.gemma = AutoModel.from_pretrained(
            "google/gemma-3-1b-it", 
            trust_remote_code=True,
            attn_implementation="eager" 
        )
        
        # Enable LoRA
        peft_config = LoraConfig(
            r=16, lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )
        self.gemma = get_peft_model(self.gemma, peft_config)
        
        # Feature Projector (Map 1792 -> 2048 Gemma Dim)
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
        # Use get_input_embeddings() to avoid AttributeError on architecture differences
        embedding_layer = self.gemma.get_input_embeddings()
        text_embeds = embedding_layer(input_ids)
        
        # IMPORTANT: Gemma models expect embeddings to be scaled by sqrt(hidden_size).
        # When passing inputs_embeds directly, this scaling is often skipped by the model,
        # so we must apply it manually.
        text_embeds = text_embeds * (self.embed_dim ** 0.5)
        
        # 2. Project the Fusion Vector (NLLB+LaBSE)
        # Shape: [Batch, 1, Hidden_Dim]
        vector_embeds = self.projector(vectors.to(dtype=torch.float32)).unsqueeze(1)
        
        # 3. Concatenate: [Vector_Token, Text_Tokens...]
        inputs_embeds = torch.cat([vector_embeds, text_embeds], dim=1)
        
        # 4. Adjust Attention Mask (Add 1 for the vector token)
        # Shape: [Batch, 1] of ones
        ones = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        extended_attention_mask = torch.cat([ones, attention_mask], dim=1)
        
        # 5. Pass to Gemma
        outputs = self.gemma(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
        
        # 6. Classification on the LAST token
        # Note: We take the last token of the sequence (which is usually the end of the text)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return self.classifier(last_hidden)

def create_dataloader(df, embeddings, labels=None, tokenizer=None, shuffle=False, batch_size=32):
    print("   -> Tokenizing Text...")
    # Tokenize the sentences
    encodings = tokenizer(
        df['Sentence'].tolist(), 
        padding=True, 
        truncation=True, 
        max_length=64, # Keep context short to save memory
        return_tensors="pt"
    )
    
    ids = encodings['input_ids']
    mask = encodings['attention_mask']
    
    if labels is not None:
        dataset = TensorDataset(ids, mask, embeddings, labels)
    else:
        dataset = TensorDataset(ids, mask, embeddings)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def apply_romanization(df):
    print("   -> 🔠 Romanizing text (Script -> Latin)...")
    
    def _transliterate_row(row):
        text = str(row['Sentence'])
        lang = row['language'] # Use the raw language name from CSV
        
        try:
            # SANTALI & MANIPURI (Usually Bengali Script or Meitei Mayek)
            # We target 'Bengali' script as source based on your dataset likelyhood
            # If your Manipuri is in Meitei Mayek, use 'MeeteiMayek' as source
            if lang in ['Santali', 'Manipuri']: 
                # Attempt conversion from Bengali script to ISO (Latin)
                # If the text is not Bengali script, it returns original usually
                return transliterate.process('Bengali', 'ISO', text)
            
            # KASHMIRI (Perso-Arabic)
            elif lang == 'Kashmiri':
                # Convert from Urdu (closest mapping for Perso-Arabic) to ISO
                return transliterate.process('Urdu', 'ISO', text)
                
        except Exception:
            return text # Fallback: keep original
            
        return text

    # Apply to the dataframe
    df['Sentence'] = df.apply(_transliterate_row, axis=1)
    return df

def main():
    print("=== STEP 2: Training (Hybrid: Text + Fusion + Mixup) ===")
    
    if not os.path.exists(config.TRAIN_EMBEDS):
        print("❌ Error: Fused embeddings not found. Run embed_fusion.py first.")
        return

    # 1. Initialize Tokenizer (Needed for Data Loading)
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    # 2. Load Embeddings
    print("Loading Vectors...")
    X_train = torch.load(config.TRAIN_EMBEDS)
    X_val = torch.load(config.VAL_EMBEDS)
    X_test = torch.load(config.TEST_EMBEDS)
    
    # 3. Load CSVs
    print("Loading CSV Data...")
    df_train = pd.read_csv(config.TRAIN_CSV)
    df_val = pd.read_csv(config.VAL_CSV)
    df_test = pd.read_csv(config.TEST_CSV)
    
    df_train = apply_romanization(df_train)
    df_val = apply_romanization(df_val)
    df_test = apply_romanization(df_test)

    # Filter Languages
    df_train = df_train[df_train['language'].isin(config.LANG_MAP.keys())]
    df_val = df_val[df_val['language'].isin(config.LANG_MAP.keys())]
    
    # Prepare Labels
    y_train = torch.tensor(df_train['emotion'].map(config.EMOTION_MAP).values, dtype=torch.long)
    y_val = torch.tensor(df_val['emotion'].map(config.EMOTION_MAP).values, dtype=torch.long)
    
    # 4. Create DataLoaders (Now includes Text + Vectors)
    print("Creating DataLoaders...")
    train_loader = create_dataloader(df_train, X_train, y_train, tokenizer, shuffle=True, batch_size=config.BATCH)
    val_loader = create_dataloader(df_val, X_val, y_val, tokenizer, shuffle=False, batch_size=config.BATCH)
    
    # 5. Initialize Model
    model = HybridGemmaClassifier(input_dim=config.INPUT_DIM).to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    EPOCHS = config.EPOCH
    best_macro_f1 = 0.0
    BEST_MODEL_PATH = "best_hybrid.pth"
    
    print(f"\nStarting Training for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for b_ids, b_mask, b_vec, b_lbl in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            b_ids = b_ids.to(config.DEVICE)
            b_mask = b_mask.to(config.DEVICE)
            b_vec = b_vec.to(config.DEVICE)
            b_lbl = b_lbl.to(config.DEVICE)
            
            # --- HYBRID MIXUP STRATEGY ---
            # We mix the VECTORS, but we keep the TEXT of Sample A.
            # This forces the model to rely on the Vector for nuance if the Text is ambiguous.
            mixed_vec, targets_a, targets_b, lam, idx = mixup_data(b_vec, b_lbl, alpha=0.4)
            
            optimizer.zero_grad()
            
            # Forward pass: Text (original) + Vectors (mixed)
            outputs = model(b_ids, b_mask, mixed_vec)
            
            # Loss calculation
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b_ids, b_mask, b_vec, b_lbl in val_loader:
                b_ids = b_ids.to(config.DEVICE)
                b_mask = b_mask.to(config.DEVICE)
                b_vec = b_vec.to(config.DEVICE)
                
                outputs = model(b_ids, b_mask, b_vec)
                preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                targets.extend(b_lbl.numpy())
        
        macro_f1 = f1_score(targets, preds, average='macro')
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🌟 New Best Model! Saved to {BEST_MODEL_PATH}")

    # ==========================================
    # FINAL REPORT & SUBMISSION
    # ==========================================
    print("\nLoading Best Model...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    # Validation Report
    final_preds, final_targets = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_vec, b_lbl in val_loader:
            b_ids = b_ids.to(config.DEVICE)
            b_mask = b_mask.to(config.DEVICE)
            b_vec = b_vec.to(config.DEVICE)
            
            outputs = model(b_ids, b_mask, b_vec)
            final_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            final_targets.extend(b_lbl.numpy())

    print("\n--- Validation Report ---")
    print(classification_report(final_targets, final_preds, target_names=list(config.EMOTION_MAP.keys()), digits=4))

    # Submission
    print("\nGenerating Submission...")
    test_loader = create_dataloader(df_test, X_test, None, tokenizer, shuffle=False, batch_size=16)
    all_preds = []
    
    with torch.no_grad():
        for b_ids, b_mask, b_vec in tqdm.tqdm(test_loader, desc="Submission"):
            b_ids = b_ids.to(config.DEVICE)
            b_mask = b_mask.to(config.DEVICE)
            b_vec = b_vec.to(config.DEVICE)
            
            outputs = model(b_ids, b_mask, b_vec)
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            
    pd.DataFrame({
        'id': df_test['id'],
        'emotion': [config.ID_TO_EMOTION[p] for p in all_preds]
    }).to_csv(config.SUBMISSION_FILE, index=False)
    print(f"✅ Submission saved to {config.SUBMISSION_FILE}")

if __name__ == "__main__":
    main()