import torch
import pandas as pd
import os
import gc
from transformers import AutoTokenizer, AutoModel
import config

# Use NLLB-200 Distilled (Fast, High Performance, 1024-dim)
MODEL_NAME = "facebook/nllb-200-distilled-600M"

def get_nllb_embeddings():
    print("=== STEP 1: NLLB Embedding Generation ===")
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(config.DEVICE)
    model.eval()

    def generate(csv_path, out_path, filter_langs=True):
        if not os.path.exists(csv_path): return
        print(f"Processing {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
        # Filter Languages
        if filter_langs:
            df = df[df['language'].isin(config.LANG_MAP.keys())]
        
        # Map to NLLB codes (Same as SONAR codes!)
        df['lang_code'] = df['language'].map(config.LANG_MAP)
        
        all_embeddings = [None] * len(df)
        
        # Process by Language Group
        for lang_code, group in df.groupby('lang_code'):
            if pd.isna(lang_code): continue
            
            sentences = group['Sentence'].tolist()
            indices = group.index.tolist()
            
            # NLLB requires setting the source language
            tokenizer.src_lang = lang_code
            
            # Batching to save memory
            BATCH_SIZE = 32
            for i in range(0, len(sentences), BATCH_SIZE):
                batch_text = sentences[i : i+BATCH_SIZE]
                batch_idx = indices[i : i+BATCH_SIZE]
                
                inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.DEVICE)
                
                with torch.no_grad():
                    # Encoder-only inference
                    outputs = model.encoder(**inputs)
                    
                    # Mean Pooling (Attention-masked)
                    mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    mean_embeddings = sum_embeddings / sum_mask
                    
                    # Normalize (Improves Classification)
                    mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
                    
                    # Move to CPU
                    mean_embeddings = mean_embeddings.cpu()
                
                # Store results
                for k, result_idx in enumerate(batch_idx):
                    all_embeddings[result_idx] = mean_embeddings[k]

        # Clean up list
        final_list = [e for e in all_embeddings if e is not None]
        final_tensor = torch.stack(final_list)
        
        torch.save(final_tensor, out_path)
        print(f"Saved {out_path} (Shape: {final_tensor.shape})")

    # Run
    generate(config.TRAIN_CSV, config.TRAIN_EMBEDS, True)
    generate(config.VAL_CSV, config.VAL_EMBEDS, True)
    generate(config.TEST_CSV, config.TEST_EMBEDS, False)
    
    print("✅ Done! Now run train.py")

if __name__ == "__main__":
    get_nllb_embeddings()