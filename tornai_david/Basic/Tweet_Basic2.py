# ==============================================================================
# 1. ALAPVETŐ IMPORTÁLÁSOK
# ==============================================================================

import os
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 2. MODULOK IMPORTÁLÁSA
# ==============================================================================

# Scikit-learn (mindig szükséges)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    sklearn_available = True
    print("✓ Scikit-learn importálva")
except ImportError as e:
    print(f"✗ Scikit-learn hiba: {e}")
    sklearn_available = False

# PyTorch (opcionális)
torch_available = False
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_available = True
    print(f"✓ PyTorch {torch.__version__} - {device}")
except ImportError:
    print("✗ PyTorch nem elérhető")
    device = 'cpu'

# Transformers (opcionális)
transformers_available = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer
    from datasets import Dataset
    transformers_available = True
    print("✓ Transformers importálva")
except ImportError:
    print("✗ Transformers nem elérhető")

# Sentence Transformers (opcionális)
sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
    print("✓ Sentence-transformers importálva")
except ImportError:
    print("✗ Sentence-transformers nem elérhető")

# Reprodukálhatóság
random.seed(42)
np.random.seed(42)
if torch_available:
    torch.manual_seed(42)

# ==============================================================================
# 3. ADATOK BETÖLTÉSE
# ==============================================================================

def load_data():
    """Adatok betöltése különböző útvonalakról"""
    possible_paths = [
        ('/kaggle/input/nlp-getting-started/train.csv', '/kaggle/input/nlp-getting-started/test.csv'),
        ('/kaggle/input/train.csv', '/kaggle/input/test.csv'),
        ('train.csv', 'test.csv')
    ]
    
    for train_path, test_path in possible_paths:
        try:
            if os.path.exists(train_path) and os.path.exists(test_path):
                df_train = pd.read_csv(train_path)
                df_test = pd.read_csv(test_path)
                
                print(f"\n✓ Adatok betöltve: {train_path}")
                print(f"Train shape: {df_train.shape}")
                print(f"Test shape: {df_test.shape}")
                
                # Adattisztítás
                df_train['text'] = df_train['text'].fillna('').astype(str)
                df_test['text'] = df_test['text'].fillna('').astype(str)
                
                # Üres szövegek kezelése
                df_train = df_train[df_train['text'].str.len() > 0]
                
                print(f"Címke eloszlás: {df_train['target'].value_counts().to_dict()}")
                print(f"Átlagos szöveghossz: {df_train['text'].str.len().mean():.1f}")
                
                return df_train, df_test
                
        except Exception as e:
            print(f"Hiba {train_path}: {e}")
            continue
    
    print("✗ Adatok nem találhatók!")
    return None, None

# Adatok betöltése
df_train, df_test = load_data()

if df_train is None:
    print("Adatok hiányoznak, demo adatok generálása...")
    # Demo adatok
    demo_texts = [
        "Building collapsed after earthquake",
        "Beautiful sunset at the beach",
        "Flood warning issued for downtown",
        "Happy birthday celebration",
        "Wildfire spreading rapidly",
        "Enjoying coffee this morning"
    ]
    demo_labels = [1, 0, 1, 0, 1, 0]
    
    df_train = pd.DataFrame({
        'id': range(len(demo_texts)),
        'text': demo_texts,
        'target': demo_labels
    })
    
    df_test = pd.DataFrame({
        'id': range(100, 110),
        'text': demo_texts[:10] if len(demo_texts) >= 10 else demo_texts * 2
    })
    
    print(f"Demo adatok: {df_train.shape}, {df_test.shape}")

# ==============================================================================
# 4. TF-IDF BASELINE MODELL
# ==============================================================================

def tfidf_model(df_train, df_test):
    """TF-IDF + Logistic Regression baseline"""
    if not sklearn_available:
        print("Scikit-learn nem elérhető")
        return None, 0.0
    
    print("\n--- TF-IDF Baseline ---")
    
    try:
        X = df_train['text'].values
        y = df_train['target'].values
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        
        # Logistic Regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        
        # Validáció
        y_pred = model.predict(X_val_tfidf)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        print(f"F1-score: {f1:.4f}")
        print(classification_report(y_val, y_pred))
        
        # Test predikció
        X_test_tfidf = vectorizer.transform(df_test['text'].values)
        test_pred = model.predict(X_test_tfidf)
        
        return test_pred, f1
        
    except Exception as e:
        print(f"TF-IDF hiba: {e}")
        return None, 0.0

# ==============================================================================
# 6. SENTENCE TRANSFORMER MODELL
# ==============================================================================

def sentence_transformer_model(df_train, df_test):
    """Sentence Transformer + Classifier"""
    if not sentence_transformers_available or not sklearn_available:
        print("Sentence transformers vagy sklearn nem elérhető")
        return None, 0.0
    
    print("\n--- Sentence Transformer ---")
    
    try:
        # Kompakt modell
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        X = df_train['text'].values
        y = df_train['target'].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Embeddings
        print("Embeddings számítása...")
        train_emb = model.encode(X_train, show_progress_bar=True)
        val_emb = model.encode(X_val, show_progress_bar=True)
        
        # Classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(train_emb, y_train)
        
        # Validáció
        y_pred = classifier.predict(val_emb)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        print(f"F1-score: {f1:.4f}")
        print(classification_report(y_val, y_pred))
        
        # Test predikció
        test_emb = model.encode(df_test['text'].values, show_progress_bar=True)
        test_pred = classifier.predict(test_emb)
        
        return test_pred, f1
        
    except Exception as e:
        print(f"Sentence transformer hiba: {e}")
        return None, 0.0

# ==============================================================================
# 7. EGYSZERŰ TRANSFORMER MODELL
# ==============================================================================

def simple_transformer_model(df_train, df_test):
    """Egyszerű transformer fine-tuning"""
    if not transformers_available or not torch_available:
        print("Transformers vagy PyTorch nem elérhető")
        return None, 0.0
    
    print("\n--- Transformer Fine-tuning ---")
    
    try:
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        
        # Adatok előkészítése
        X = df_train['text'].tolist()
        y = df_train['target'].tolist()
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Tokenizálás
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                padding='max_length', 
                truncation=True, 
                max_length=128
            )
        
        train_dataset = Dataset.from_dict({'text': X_train, 'labels': y_train})
        val_dataset = Dataset.from_dict({'text': X_val, 'labels': y_val})
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Training argumentumok
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            load_best_model_at_end=True,
            report_to="none",
            save_strategy="epoch",
            save_total_limit=1
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'f1': f1_score(labels, predictions, average='weighted')}
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Training
        print("Modell tanítása...")
        trainer.train()
        
        # Validáció
        eval_results = trainer.evaluate()
        f1 = eval_results['eval_f1']
        print(f"F1-score: {f1:.4f}")
        
        # Test predikció
        test_dataset = Dataset.from_dict({'text': df_test['text'].tolist()})
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        predictions = trainer.predict(test_dataset)
        test_pred = np.argmax(predictions.predictions, axis=1)
        
        return test_pred, f1
        
    except Exception as e:
        print(f"Transformer hiba: {e}")
        return None, 0.0

# ==============================================================================
# 8. MODELLEK FUTTATÁSA
# ==============================================================================

results = {}
final_predictions = None
best_f1 = 0.0

# 1. TF-IDF (baseline)
tfidf_pred, tfidf_f1 = tfidf_model(df_train, df_test)
if tfidf_pred is not None:
    results['TF-IDF'] = tfidf_f1
    if tfidf_f1 > best_f1:
        final_predictions = tfidf_pred
        best_f1 = tfidf_f1

# 2. Sentence Transformer
st_pred, st_f1 = sentence_transformer_model(df_train, df_test)
if st_pred is not None:
    results['Sentence Transformer'] = st_f1
    if st_f1 > best_f1:
        final_predictions = st_pred
        best_f1 = st_f1

# 3. Transformer Fine-tuning
trans_pred, trans_f1 = simple_transformer_model(df_train, df_test)
if trans_pred is not None:
    results['Transformer'] = trans_f1
    if trans_f1 > best_f1:
        final_predictions = trans_pred
        best_f1 = trans_f1

# ==============================================================================
# 9. EREDMÉNYEK ÉS SUBMISSION
# ==============================================================================

print("\n" + "="*50)
print("VÉGSŐ EREDMÉNYEK")
print("="*50)

if results:
    for model, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:<20}: {f1:.4f}")
    
    print(f"\nLegjobb modell F1: {best_f1:.4f}")
else:
    print("Nincs sikeres modell")

# Submission fájl
if final_predictions is not None:
    submission = pd.DataFrame({
        'id': df_test['id'],
        'target': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    print(f"\nSubmission fájl elkészítve!")
    print(f"Predikciók: {pd.Series(final_predictions).value_counts().to_dict()}")
else:
    print("\nNincs predikció a submission fájlhoz")

print("="*50)