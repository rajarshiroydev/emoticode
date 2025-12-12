# embed_fusion.py
import torch
import pandas as pd
import os
import gc
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import config

def get_fused_embeddings():
    print("=== STEP 1: Feature Fusion (NLLB + LaBSE) ===")
    
    # --- MODEL 1: NLLB (1024 dim) ---
    print("1. Generatng NLLB Embeddings...")
    nllb_model_name = "facebook/nllb-200-distilled-600M"
    nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
    nllb_model = AutoModel.from_pretrained(nllb_model_name).to(config.DEVICE)
    nllb_model.eval()

    # --- MODEL 2: LaBSE (768 dim) ---
    print("2. Generating LaBSE Embeddings...")
    labse_model = SentenceTransformer("sentence-transformers/LaBSE", device=config.DEVICE)

    def generate(csv_path, out_path, filter_langs=True):
        if not os.path.exists(csv_path): return
        print(f"   Processing {csv_path}...")
        
        df = pd.read_csv(csv_path)
        if filter_langs:
            df = df[df['language'].isin(config.LANG_MAP.keys())]
        
        # IMPORTANT: Reset index to ensure alignment
        df.reset_index(drop=True, inplace=True)
        sentences = df['Sentence'].tolist()
        df['lang_code'] = df['language'].map(config.LANG_MAP)

        # --- PART A: Get LaBSE Vectors (Bulk) ---
        print("     -> Encoding LaBSE...")
        labse_emb = labse_model.encode(sentences, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
        # Ensure it's on CPU
        labse_emb = labse_emb.cpu()

        # --- PART B: Get NLLB Vectors (By Language) ---
        print("     -> Encoding NLLB...")
        nllb_emb_list = [None] * len(df)
        
        for lang_code, group in df.groupby('lang_code'):
            if pd.isna(lang_code): continue
            
            grp_sentences = group['Sentence'].tolist()
            grp_indices = group.index.tolist()
            
            nllb_tokenizer.src_lang = lang_code
            
            BATCH_SIZE = 32
            for i in range(0, len(grp_sentences), BATCH_SIZE):
                batch_text = grp_sentences[i : i+BATCH_SIZE]
                batch_idx = grp_indices[i : i+BATCH_SIZE]
                
                inputs = nllb_tokenizer(batch_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(config.DEVICE)
                
                with torch.no_grad():
                    outputs = nllb_model.encoder(**inputs)
                    # Mean Pooling
                    mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    mean_emb = sum_emb / sum_mask
                    mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
                
                for k, idx in enumerate(batch_idx):
                    nllb_emb_list[idx] = mean_emb[k].cpu()
        
        # Handle case where NLLB might miss rows (unlikely but safe)
        nllb_tensor_list = [e if e is not None else torch.zeros(1024) for e in nllb_emb_list]
        nllb_tensor = torch.stack(nllb_tensor_list)

        # --- PART C: FUSION ---
        print("     -> Fusing Vectors...")
        # Concatenate: [Batch, 1024] + [Batch, 768] = [Batch, 1792]
        fused_tensor = torch.cat([nllb_tensor, labse_emb], dim=1)
        
        torch.save(fused_tensor, out_path)
        print(f"     -> Saved {out_path} (Shape: {fused_tensor.shape})")

    generate(config.TRAIN_CSV, config.TRAIN_EMBEDS, True)
    generate(config.VAL_CSV, config.VAL_EMBEDS, True)
    generate(config.TEST_CSV, config.TEST_EMBEDS, False)
    
    print("✅ Fusion Complete. Run train_mixup.py")

if __name__ == "__main__":
    get_fused_embeddings()