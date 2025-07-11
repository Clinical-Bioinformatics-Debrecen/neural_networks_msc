# ==============================================================================
# 1. SZÜKSÉGES CSOMAGOK IMPORTÁLÁSA
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import re
import warnings
import gc
import os
import shutil
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. KONFIGURÁCIÓ ÉS KÖRNYEZETI BEÁLLÍTÁSOK
# ==============================================================================

# Reprodukálhatóság biztosítása
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Determinisztikus CUDA műveletek
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

print("PyTorch verzió:", torch.__version__)
print("GPU elérhető:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Használt eszköz: {device}")

# Dinamikus batch size beállítása
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    batch_size = 32 if gpu_memory > 15 else 16 # T4 GPU-ra optimalizálva
    print(f"GPU memória: {gpu_memory:.1f} GB, batch size: {batch_size}")
else:
    batch_size = 8
    print(f"CPU használat, batch size: {batch_size}")

class Config:
    K_FOLDS = 3
    MODEL_NAME_TWITTER = 'cardiffnlp/twitter-roberta-base-sentiment'
    MODEL_NAME_GENERIC = 'microsoft/deberta-v3-base'
    RANDOM_STATE = 42
    BATCH_SIZE = batch_size # Dinamikus batch méret használata
    TRAIN_EPOCHS = 2
    MAX_LENGTH = 128
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

# ==============================================================================
# 1. SZÖVEG ELŐFELDOLGOZÓ OSZTÁLY
# ==============================================================================
class TextPreprocessor:
    def clean_text(self, text, keep_hashtag=False):
        if pd.isna(text): return ""
        text = str(text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'@\w+', '', text)
        if not keep_hashtag:
            text = re.sub(r'#', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_manual_features(self, texts_df):
        features = pd.DataFrame()
        features['word_count'] = texts_df.apply(lambda x: len(str(x).split()))
        features['char_count'] = texts_df.apply(lambda x: len(str(x)))
        features['unique_word_ratio'] = texts_df.apply(lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1))
        # Új feature: felkiáltójel és kérdőjel számok
        features['exclamation_count'] = texts_df.apply(lambda x: str(x).count('!'))
        features['question_count'] = texts_df.apply(lambda x: str(x).count('?'))
        features['upper_ratio'] = texts_df.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
        return features

# ==============================================================================
# 2. KLASSZIKUS ENSEMBLE MODELL OSZTÁLY
# ==============================================================================
class ClassicEnsembleModel:
    def __init__(self, config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        # Inicializálás try-except blokkban
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except Exception as e:
            print(f"Hiba az embedding modell betöltésében: {e}")
            raise
        self.scaler_manual_features = StandardScaler()

    def create_features(self, df, fit_scalers=False):
        clean_texts = df['text'].apply(lambda x: self.preprocessor.clean_text(x, keep_hashtag=False))
        
        # Üres szövegek kezelése
        clean_texts = clean_texts.fillna("empty text")
        
        # Embeddings generálása batch-ekben a memória hatékonyság érdekében
        embeddings = self.embedding_model.encode(
            clean_texts.tolist(), 
            show_progress_bar=True,
            batch_size=64  # Kisebb batch méret a memória kezeléshez
        )
        
        manual_features = self.preprocessor.extract_manual_features(clean_texts)

        if fit_scalers:
            manual_features_scaled = self.scaler_manual_features.fit_transform(manual_features)
        else:
            manual_features_scaled = self.scaler_manual_features.transform(manual_features)
            
        return np.hstack([embeddings, manual_features_scaled])

    def train_and_predict_kfold(self, df_train, df_test):
        print("\n--- Klasszikus Ensemble Modell K-Fold Tanítása ---")
        y = df_train['target'].values
        X = self.create_features(df_train, fit_scalers=True)
        X_test = self.create_features(df_test, fit_scalers=False)

        skf = StratifiedKFold(n_splits=self.config.K_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        oof_probas = np.zeros((len(df_train), 2))
        test_probas = np.zeros((len(df_test), 2))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Classic Ensemble - FOLD {fold + 1}")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Hiperparaméterek optimalizálása
            models = [
                ('lr', LogisticRegression(
                    random_state=self.config.RANDOM_STATE, 
                    max_iter=1000,
                    C=1.0,
                    solver='lbfgs'
                )),
                ('lgbm', lgb.LGBMClassifier(
                    random_state=self.config.RANDOM_STATE, 
                    verbose=-1,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6
                )),
                ('xgb', xgb.XGBClassifier(
                    random_state=self.config.RANDOM_STATE, 
                    eval_metric='logloss', 
                    verbosity=0,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6
                ))
            ]
            
            ensemble_model = VotingClassifier(estimators=models, voting='soft')
            ensemble_model.fit(X_train_fold, y_train_fold)
            
            # Validációs eredmények
            val_pred = ensemble_model.predict(X_val_fold)
            val_f1 = f1_score(y_val_fold, val_pred, average='weighted')
            print(f"Fold {fold + 1} validációs F1: {val_f1:.4f}")
            
            oof_probas[val_idx] = ensemble_model.predict_proba(X_val_fold)
            test_probas += ensemble_model.predict_proba(X_test) / self.config.K_FOLDS
        
        # Összesített validációs eredmények
        oof_pred = np.argmax(oof_probas, axis=1)
        overall_f1 = f1_score(y, oof_pred, average='weighted')
        print(f"Összesített validációs F1: {overall_f1:.4f}")
        
        return oof_probas, test_probas

# ==============================================================================
# 3. TRANSFORMER MODELL OSZTÁLY
# ==============================================================================
class TransformerKFoldModel:
    def __init__(self, config):
        self.config = config
        self.preprocessor = TextPreprocessor()

    def _compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        predictions = np.argmax(preds, axis=1)
        f1 = f1_score(labels, predictions, average="weighted")
        return {'f1': f1}

    def train_and_predict(self, df_train, df_test, model_name, keep_hashtag):
        print(f"\n--- Transformer Modell ({model_name}) Tanítása ---")
        print(f"Hashtag megtartása: {keep_hashtag}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None: 
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Hiba a tokenizer betöltésében: {e}")
            raise
            
        df_train_hf = df_train.copy()
        df_train_hf['labels'] = df_train_hf['target']
        df_train_hf['text'] = df_train_hf['text'].apply(
            lambda x: self.preprocessor.clean_text(x, keep_hashtag=keep_hashtag)
        )
        
        # Üres szövegek kezelése
        df_train_hf['text'] = df_train_hf['text'].fillna("empty text")
        
        skf = StratifiedKFold(n_splits=self.config.K_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        oof_logits = np.zeros((len(df_train), 2))
        test_logits = np.zeros((len(df_test), 2))

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_hf, df_train_hf['labels'])):
            print(f"Transformer - FOLD {fold + 1}")
            
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=2, 
                    ignore_mismatched_sizes=True
                )
                model.to(device)
            except Exception as e:
                print(f"Hiba a modell betöltésében: {e}")
                raise
            
            train_data = df_train_hf.iloc[train_idx].reset_index(drop=True)
            val_data = df_train_hf.iloc[val_idx].reset_index(drop=True)

            train_dataset = Dataset.from_pandas(train_data[['text', 'labels']])
            val_dataset = Dataset.from_pandas(val_data[['text', 'labels']])
            
            def tokenize(batch): 
                return tokenizer(
                    batch['text'], 
                    truncation=True, 
                    padding=True,
                    max_length=self.config.MAX_LENGTH
                )
            
            train_dataset = train_dataset.map(tokenize, batched=True)
            val_dataset = val_dataset.map(tokenize, batched=True)
            
            # Oszlopok eltávolítása a map után
            train_dataset = train_dataset.remove_columns(['text'])
            val_dataset = val_dataset.remove_columns(['text'])

            output_dir = f'./results_fold_{fold+1}'
            args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.TRAIN_EPOCHS,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                per_device_eval_batch_size=self.config.BATCH_SIZE,
                learning_rate=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                report_to="none",
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=False,  # Memória optimalizálás
                gradient_checkpointing=True,   # Memória takarékosság
                warmup_ratio=0.1,
                logging_steps=50
            )
            
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=self._compute_metrics
            )
            
            try:
                trainer.train()
                
                # Validációs logits
                val_predictions = trainer.predict(val_dataset)
                oof_logits[val_idx] = val_predictions.predictions
                
                # Validációs F1 score
                val_pred_labels = np.argmax(val_predictions.predictions, axis=1)
                val_f1 = f1_score(val_data['labels'], val_pred_labels, average='weighted')
                print(f"Fold {fold + 1} validációs F1: {val_f1:.4f}")
                
                # Test előrejelzések
                df_test_hf = df_test.copy()
                df_test_hf['text'] = df_test_hf['text'].apply(
                    lambda x: self.preprocessor.clean_text(x, keep_hashtag=keep_hashtag)
                )
                df_test_hf['text'] = df_test_hf['text'].fillna("empty text")
                
                test_dataset_hf = Dataset.from_pandas(df_test_hf[['text']])
                test_dataset_tokenized = test_dataset_hf.map(tokenize, batched=True)
                test_dataset_tokenized = test_dataset_tokenized.remove_columns(['text'])
                
                test_predictions = trainer.predict(test_dataset_tokenized)
                test_logits += test_predictions.predictions / self.config.K_FOLDS
                
            except Exception as e:
                print(f"Hiba a fold {fold + 1} tanításában: {e}")
                raise
            finally:
                # Cleanup
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                del model, trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Összesített validációs eredmények
        final_oof_labels = np.argmax(oof_logits, axis=1)
        overall_f1 = f1_score(df_train['target'], final_oof_labels, average='weighted')
        print(f"Összesített validációs F1: {overall_f1:.4f}")
        print("\nÖsszesített validációs riport:")
        print(classification_report(df_train['target'], final_oof_labels, digits=4))
        
        return oof_logits, test_logits

# ==============================================================================
# 4. ADATBETÖLTÉS ÉS FŐPROGRAM
# ==============================================================================
def load_and_validate_data(train_path, test_path):
    """Adatok betöltése, hiányzó értékek kezelése és validálása."""
    try:
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Hiányzó fájlok: {train_path} vagy {test_path}")
            
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        required_train_cols = ['text', 'target']
        required_test_cols = ['text', 'id']
        
        if not all(col in df_train.columns for col in required_train_cols):
            raise ValueError(f"Hiányzó oszlopok a train adatokban: {required_train_cols}")
        
        if not all(col in df_test.columns for col in required_test_cols):
            raise ValueError(f"Hiányzó oszlopok a test adatokban: {required_test_cols}")
        
        initial_len = len(df_train)
        df_train = df_train.dropna(subset=required_train_cols)
        if len(df_train) < initial_len:
            print(f"Figyelmeztetés: {initial_len - len(df_train)} sor eltávolítva hiányzó értékek miatt.")
        
        # Szövegek string-re konvertálása
        df_train['text'] = df_train['text'].astype(str)
        df_test['text'] = df_test['text'].astype(str)
        
        # Target validálás
        if not df_train['target'].isin([0, 1]).all():
            raise ValueError("A 'target' oszlop nem csak 0 és 1 értékeket tartalmaz.")
        
        # Üres szövegek kezelése
        df_test['text'] = df_test['text'].fillna('empty text')
        
        # Osztály eloszlás ellenőrzése
        class_counts = df_train['target'].value_counts()
        print(f"Osztály eloszlás: {class_counts.to_dict()}")
        
        print(f"Adatok sikeresen betöltve. Train: {len(df_train)}, Test: {len(df_test)}")
        return df_train, df_test
        
    except Exception as e:
        print(f"Hiba az adatok betöltésében: {e}")
        return None, None

def safe_log_odds(probas, epsilon=1e-15):
    """Biztonságos log-odds konverzió."""
    probas = np.clip(probas, epsilon, 1-epsilon)
    return np.log(probas / (1 - probas))

def main():
    config = Config()
    
    # Adatok betöltése
    df_train, df_test = load_and_validate_data(
        "../input/nlp-getting-started/train.csv", 
        "../input/nlp-getting-started/test.csv"
    )
    if df_train is None or df_test is None:
        print("Adatok betöltése sikertelen.")
        return

    # Klasszikus ensemble modell
    try:
        classic_model = ClassicEnsembleModel(config)
        classic_oof_probas, classic_test_probas = classic_model.train_and_predict_kfold(df_train, df_test)
        
        # Biztonságos log-odds konverzió
        classic_oof_logits = safe_log_odds(classic_oof_probas)
        classic_test_logits = safe_log_odds(classic_test_probas)
        
    except Exception as e:
        print(f"Hiba a klasszikus modellben: {e}")
        return

    # Transformer modell
    try:
        transformer_runner = TransformerKFoldModel(config)
        twitter_oof_logits, twitter_test_logits = transformer_runner.train_and_predict(
            df_train, df_test, 
            model_name=config.MODEL_NAME_TWITTER, 
            keep_hashtag=True
        )
    except Exception as e:
        print(f"Hiba a transformer modellben: {e}")
        return

    # Meta-ensemble súlyok optimalizálása
    print("\n--- Meta-Ensemble súlyok validálása ---")
    best_f1, best_weight = 0, 0
    
    for w in np.arange(0, 1.01, 0.1):
        try:
            combined_logits = w * classic_oof_logits + (1 - w) * twitter_oof_logits
            preds = np.argmax(combined_logits, axis=1)
            f1 = f1_score(df_train['target'].values, preds, average='weighted')
            print(f"Súly (Klasszikus: {w:.1f}, Transformer: {1-w:.1f}) | OOF F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1, best_weight = f1, w
                
        except Exception as e:
            print(f"Hiba a súly {w:.1f} értéknél: {e}")
            continue
    
    print(f"\nLegjobb súly a klasszikus modellhez: {best_weight:.1f}")
    print(f"Legjobb OOF F1 pontszám: {best_f1:.4f}")

    # Végső előrejelzések
    try:
        final_logits = best_weight * classic_test_logits + (1 - best_weight) * twitter_test_logits
        final_predictions = np.argmax(final_logits, axis=1)
        
        # Ellenőrizni, hogy az előrejelzések érvényesek
        if not np.all(np.isin(final_predictions, [0, 1])):
            print("Figyelem: Az előrejelzések nem csak 0 és 1 értékeket tartalmaznak!")
        
        # Submission file létrehozása
        submission_df = pd.DataFrame({
            'id': df_test['id'], 
            'target': final_predictions
        })
        
        submission_df.to_csv('submission_meta_ensemble_optimized.csv', index=False)
        print(f"\nOptimalizált meta-ensemble submission fájl létrehozva.")
        print(f"Előrejelzések eloszlása: {pd.Series(final_predictions).value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Hiba a végső submission létrehozásában: {e}")

if __name__ == "__main__":
    main()