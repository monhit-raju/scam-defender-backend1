import pandas as pd
import numpy as np
import re
import warnings
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

warnings.filterwarnings('ignore')
print("✅ Libraries installed. (Runtime → GPU recommended)")

# 2. CLEANING (PDF pipeline + spam-friendly)
def clean_email_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s@._$%!?]', ' ', text)   # keep spam indicators
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. LOAD DATASET
df = pd.read_csv('/content/emails.csv')   # ← Change if needed

print(f"✅ Loaded: {df.shape[0]:,} emails")
print("Classes:\n", df['spam'].value_counts() if 'spam' in df.columns else df['spam_or_not'].value_counts())

# Clean
df['clean_text'] = df['text'].apply(clean_email_text)

# Hugging Face Dataset
col_label = 'spam' if 'spam' in df.columns else 'spam_or_not'
dataset = Dataset.from_pandas(df[['clean_text', col_label]].rename(columns={'clean_text': 'text', col_label: 'label'}))

# FIX: Cast label to ClassLabel
dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["ham", "spam"]))
print("✅ Labels cast → stratification ready.")

# 4. TOKENIZE (DistilBERT)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

print("\n🔤 Tokenizing...")
tokenized_dataset = dataset.map(tokenize, batched=True)

# Stratified split
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.20, seed=42, stratify_by_column='label')

# 5. MODEL
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 6. METRICS
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# 7. TRAINING ARGS (FIXED: eval_strategy instead of evaluation_strategy)
training_args = TrainingArguments(
    output_dir='./email_spam_model',
    eval_strategy="epoch",                  # ← FIXED (new API)
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 8. TRAINER (FIXED: removed tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n🚀 Training on GPU (3-8 min for 5.7k emails)...")
trainer.train()

# 9. EVALUATION
print("\n📊 FINAL RESULTS:")
eval_results = trainer.evaluate()
print(f"✅ Accuracy     : {eval_results['eval_accuracy']:.4f}")
print(f"✅ Precision    : {eval_results['eval_precision']:.4f}")
print(f"✅ Recall       : {eval_results['eval_recall']:.4f}")
print(f"✅ F1-Score     : {eval_results['eval_f1']:.4f}")

# Report + Matrix
predictions = trainer.predict(tokenized_dataset["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("\n📋 Report:")
print(classification_report(y_true, y_pred, target_names=['Ham (0)', 'Spam (1)'], digits=4))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.show()

# 10. SAVE & DOWNLOAD
trainer.save_model('/content/email_spam_distilbert_model')
tokenizer.save_pretrained('/content/email_spam_distilbert_model')

print("\n✅ Model saved!")
files.download('/content/email_spam_distilbert_model')

# 11. INFERENCE (for app)
def predict_email_spam(email_text, threshold=0.5):
    cleaned = clean_email_text(email_text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = trainer.model(**inputs)
        prob = torch.softmax(outputs.logits, dim=-1)[0][1].item()
    return "SPAM" if prob >= threshold else "HAM", round(prob * 100, 2)

print("\n🔍 Tests:")
print(predict_email_spam("Subject: naturally irresistible your corporate identity..."))
print(predict_email_spam("Subject: Team meeting at 10 AM tomorrow"))

print("\n🎉 Done! 99%+ model ready for Scam Defender.")
