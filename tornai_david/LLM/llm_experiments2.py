import pandas as pd
import numpy as np
import torch
import re
import logging
import gc
from typing import List, Dict, Optional
import os
import json
from datetime import datetime
from tqdm import tqdm
import random
from sklearn.metrics import f1_score

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==============================================================================
# 0. KONFIGURÁCIÓ ÉS KÖRNYEZETI BEÁLLÍTÁSOK
# ==============================================================================
# Reprodukálhatóság biztosítása
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class Config:
    OLLAMA_MODEL_LIST = ['phi3:mini', 'gemma:2b', 'qwen:1.8b']
    GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
    PROMPT_TEMPLATES = {
        "simple_yes_no": ("Answer with only 'YES' or 'NO'. Is this a real disaster? Tweet: \"{tweet_text}\""),
        "reasoning_and_classify": ("Reason step-by-step if this tweet is a real disaster, then conclude with 'Classification: DISASTER' or 'Classification: NOT DISASTER'. Tweet: \"{tweet_text}\"")
    }
    DATA_FILE = 'train.csv'
    SAMPLE_SIZE = 100
    OUTPUT_FILE = 'llm_zero_shot_results.csv'
    LOG_LEVEL = logging.INFO
    GENERATION_CONFIG = {'max_new_tokens': 60, 'temperature': 0.1}

# ==============================================================================
# 1. SEGÉDFÜGGVÉNYEK ÉS FŐFÜGGVÉNYEK
# ==============================================================================
def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """A naplózási rendszer beállítása."""
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'llm_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), 
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_and_validate_data(filepath: str, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Adatok betöltése, alapos validálása és stratifikált mintavételezés."""
    logger = logging.getLogger(__name__)
    logger.info(f"Adatok betöltése: {filepath}")
    try:
        df = pd.read_csv(filepath)
        required_cols = ['text', 'target']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Hiányzó oszlopok a train adatokban.")
        
        initial_len = len(df)
        df.dropna(subset=required_cols, inplace=True)
        if len(df) < initial_len: 
            logger.warning(f"{initial_len - len(df)} sor eltávolítva hiányzó értékek miatt.")
            
        df['text'] = df['text'].astype(str)
        if not df['target'].isin([0, 1]).all():
            raise ValueError("A 'target' oszlop nem csak 0 és 1 értékeket tartalmaz.")

        if sample_size and sample_size < len(df):
            logger.info(f"{sample_size} minta kiválasztása (stratifikált).")
            stratify_col = df['target']
            sampled_dfs = []
            
            # Javított mintavételezés
            for group_value in stratify_col.unique():
                group_df = df[df['target'] == group_value]
                sample_count = max(1, int(np.round(sample_size * len(group_df) / len(df))))
                sample_count = min(sample_count, len(group_df))
                sampled_group = group_df.sample(n=sample_count, random_state=42)
                sampled_dfs.append(sampled_group)
            
            df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Adatok sikeresen betöltve. Méret: {len(df)} sor.")
        return df
    except FileNotFoundError:
        logger.error(f"Hiba: A fájl nem található: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Váratlan hiba az adatok betöltése során: {e}")
        return None

def parse_llm_response(response: str, prompt_key: str) -> int:
    """A modell szöveges válaszának átalakítása bináris címkévé."""
    if not response: 
        return 0
    response_clean = str(response).lower().strip()
    
    if prompt_key == "simple_yes_no":
        words = response_clean.split()
        if any(word.startswith('yes') for word in words): 
            return 1
        elif any(word.startswith('no') for word in words): 
            return 0
        else: 
            return 0
    elif prompt_key == "reasoning_and_classify":
        lines = [line.strip() for line in response_clean.split('\n') if line.strip()]
        for line in reversed(lines):
            if 'classification:' in line:
                if 'not disaster' in line or 'not_disaster' in line: 
                    return 0
                elif 'disaster' in line: 
                    return 1
        # Fallback az egész válasz átvizsgálására
        if 'not disaster' in response_clean: 
            return 0
        elif 'disaster' in response_clean: 
            return 1
    return 0

def run_classification_ollama(df: pd.DataFrame, model_name: str, prompt_template: str, prompt_key: str) -> List[int]:
    """Osztályozás futtatása Ollama segítségével."""
    logger = logging.getLogger(__name__)
    if not OLLAMA_AVAILABLE: 
        raise ImportError("'ollama' könyvtár nincs telepítve.")
    logger.info(f"Modell futtatása (Ollama): {model_name}")

    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Ollama ({model_name})"):
        try:
            prompt = prompt_template.format(tweet_text=row['text'])
            response = ollama.generate(model=model_name, prompt=prompt)
            response_text = response.get('response', '') if response else ''
            predictions.append(parse_llm_response(response_text, prompt_key))
        except Exception as e:
            logger.warning(f"Hiba egy minta feldolgozása során: {e}. Alapértelmezett (0) érték használata.")
            predictions.append(0)
    return predictions

def run_classification_gemini(df: pd.DataFrame, model_name: str, prompt_template: str, prompt_key: str) -> List[int]:
    """Osztályozás futtatása Gemini segítségével."""
    logger = logging.getLogger(__name__)
    if not GEMINI_AVAILABLE: 
        raise ImportError("'google.generativeai' könyvtár nincs telepítve.")
    if not os.getenv('GOOGLE_API_KEY'): 
        raise ValueError("GOOGLE_API_KEY környezeti változó nincs beállítva.")
    
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel(model_name)
    logger.info(f"Modell futtatása (Gemini): {model_name}")

    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Gemini ({model_name})"):
        try:
            prompt = prompt_template.format(tweet_text=row['text'])
            response = model.generate_content(prompt)
            response_text = response.text if response and hasattr(response, 'text') else ''
            predictions.append(parse_llm_response(response_text, prompt_key))
        except Exception as e:
            logger.warning(f"Hiba egy minta feldolgozása során: {e}. Alapértelmezett (0) érték használata.")
            predictions.append(0)
    return predictions

# ==============================================================================
# 3. FŐPROGRAM
# ==============================================================================
def main():
    config = Config()
    logger = setup_logging(config.LOG_LEVEL)
    
    # Mód választás parancssori argumentumokkal vagy környezeti változóval
    EXECUTION_MODE = os.getenv('LLM_MODE', 'ollama')  # alapértelmezetten 'ollama'
    
    logger.info(f"LLM Zero-shot kísérlet indítása '{EXECUTION_MODE}' módban.")
    df = load_and_validate_data(config.DATA_FILE, config.SAMPLE_SIZE)
    if df is None: 
        logger.error("Nem sikerült betölteni az adatokat. Kilépés.")
        return

    true_labels = df['target'].tolist()
    results = []
    
    if EXECUTION_MODE == 'ollama':
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama nem elérhető, de az 'ollama' mód van kiválasztva.")
            return
        model_list, run_function = config.OLLAMA_MODEL_LIST, run_classification_ollama
    elif EXECUTION_MODE == 'gemini':
        if not GEMINI_AVAILABLE:
            logger.error("Gemini nem elérhető, de a 'gemini' mód van kiválasztva.")
            return
        model_list, run_function = [config.GEMINI_MODEL_NAME], run_classification_gemini
    else:
        logger.error(f"Ismeretlen végrehajtási mód: {EXECUTION_MODE}")
        return

    for model_name in model_list:
        for prompt_key, prompt_template in config.PROMPT_TEMPLATES.items():
            try:
                logger.info(f"Futtatás: {model_name} - {prompt_key}")
                predictions = run_function(df, model_name, prompt_template, prompt_key)
                
                if len(predictions) != len(true_labels):
                    logger.error("Predikciók és valós címkék hossza nem egyezik.")
                    continue
                    
                f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
                logger.info(f"Eredmény - Modell: {model_name}, Prompt: {prompt_key}, F1-Score: {f1:.4f}")
                results.append({
                    'model': model_name, 
                    'prompt_type': prompt_key, 
                    'f1_score': f1,
                    'predictions': predictions  # Predikciók tárolása további elemzéshez
                })
                gc.collect()
            except Exception as e:
                logger.error(f"Hiba a(z) {model_name} ({prompt_key}) futtatása során: {e}")

    if results:
        # Eredmények DataFrame-be mentése (predictions nélkül a CSV-hez)
        results_for_csv = [{k: v for k, v in r.items() if k != 'predictions'} for r in results]
        results_df = pd.DataFrame(results_for_csv)
        results_df.to_csv(config.OUTPUT_FILE, index=False)
        logger.info(f"Eredmények mentve: {config.OUTPUT_FILE}")
        logger.info("\nVégső eredménytábla:\n" + results_df.sort_values(by='f1_score', ascending=False).to_string(index=False))
    else:
        logger.warning("Nincsenek eredmények a mentéshez.")

if __name__ == "__main__":
    main()