import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
from sklearn.metrics import classification_report


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


df = pd.read_csv("news_sample3.csv") #Read the file and load to data frame 


dfs1 = df.dropna(subset=['type']).reset_index(drop=True)# This removes the rows where 'type' is missing and assigns to dfs1
stp_wds = set(stopwords.words('english'))# This remove the stopwords such as article and pronouns
stemmer = PorterStemmer()   # To convert the words to their basic form
stemmer = PorterStemmer()

def TokanizeOnly(text):
    if not isinstance(text, str):#check weather the input is string
        return []
    tokens = word_tokenize(text.lower())#convert all charactors to lower case
    return [word for word in tokens if word.isalpha()] #remove numbers, symbols and keep only words

dfs1['OnlyTokans'] = dfs1['content'].apply(TokanizeOnly)# Create new column 'OnlyTokans'
dfs1['no_stopwords'] = dfs1['OnlyTokans'].apply(
    lambda tokens: [word for word in tokens if word not in stp_wds]
)
dfs1['stemd'] = dfs1['no_stopwords'].apply(
    lambda tokens: [stemmer.stem(word) for word in tokens]  #Remove stop words
)#apply stemming to tokans

# Frequency of Top 10000 words vs Word Rank
processed_vocab = Counter([word for tokens in dfs1['stemd'] for word in tokens])
tp_wds = min(10000, len(processed_vocab))
TopWords = processed_vocab.most_common(tp_wds)

if TopWords:
    words, fqs = zip(*TopWords)
    plt.figure(figsize=(15, 5))
    plt.plot(fqs)
    plt.title(f"Frequency of Top {tp_wds} Words vs word rank")
    plt.xlabel("Word Rank")
    plt.ylabel("Frequency")
    plt.show()

#  Count URLs, dates, and numbers
dfs1['url_count'] = dfs1['content'].str.count(r'http[s]?://\S+')
DatePtn = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b'
dfs1['date_count'] = dfs1['content'].str.count(DatePtn, flags=re.IGNORECASE)
dfs1['number_count'] = dfs1['content'].str.count(r'\d+')

# Removal of URLs and numbers from content
def remove_urls_and_numbers(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\d+', '', text)
    return text

dfs1['content_cleaned'] = dfs1['content'].apply(remove_urls_and_numbers)

# Final token processing on cleaned content
def process_text(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stp_wds]
    return [stemmer.stem(token) for token in tokens]

# count the number of rows became empty after processing
empty_token_count = dfs1['content'].apply(lambda x: len(x) == 0).sum()
print(f"Number of rows with empty tokens: {empty_token_count}")

# Show most common words (top 20)
processed_vocab = Counter([token for tokens in dfs1['content'] for token in tokens])
print("Top 20 tokens:", processed_vocab.most_common(20))

# frequency of top 100 word vs words
processed_vocab = Counter([word for tokens in dfs1['stemd'] for word in tokens])
tp_wds = min(100, len(processed_vocab))  # Limit for visibility
TopWords = processed_vocab.most_common(tp_wds)

if TopWords:
    words, fqs = zip(*TopWords)
    plt.figure(figsize=(15, 5))
    plt.bar(words, fqs)
    plt.title(f"Frequency of Top {tp_wds} Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for frequency plot.")

df = dfs1.copy()
# Use the 'type' field as label source, clean it
df = df[df['type'].notnull()]# drop rows with no label
df['label_binary'] = df['type'].apply(
    lambda x: 'reliable' if str(x).strip().lower() in ['reliable', 'political'] else 'fake'
)
# Check class balance
print(df['label_binary'].value_counts())

# Train/Val/Test Split (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label_binary'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_binary'], random_state=42)

# Feature extraction from 'content'
Vect = TfidfVectorizer(stop_words='english', max_features=5000)
XTrain = Vect.fit_transform(train_df['content'].fillna(""))
XValue = Vect.transform(val_df['content'].fillna(""))
XTest = Vect.transform(test_df['content'].fillna(""))

YTrain = train_df['label_binary']
YValue = val_df['label_binary']
YTest = test_df['label_binary']

# Create a function "ModelEval" to evaluate a model
def ModelEval(name, model, XValue, YValue, XTest, YTest):
    print(f"\n{name} Performance:")
    for split_name, X, y in [("Validation", XValue, YValue), ("Test", XTest, YTest)]:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, pos_label='reliable', zero_division=0)
        rec = recall_score(y, y_pred, pos_label='reliable', zero_division=0)
        f1 = f1_score(y, y_pred, pos_label='reliable', zero_division=0)
        print(f"\n{split_name} Set:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")


# Naive Bayes model
NavModl = MultinomialNB()
NavModl.fit(XTrain, YTrain)
ModelEval("Naive Bayes", NavModl, XValue, YValue, XTest, YTest)

#  Logistic Regression model with class weights
LogMdl = LogisticRegression(class_weight='balanced', max_iter=1000)
LogMdl.fit(XTrain, YTrain)
ModelEval("Logistic Regression", LogMdl, XValue, YValue, XTest, YTest)

#Advance Model
df_bert = pd.read_csv("news_sample3.csv")  

df_bert = df_bert[df_bert['inserted_at'].notnull() & df_bert['content'].notnull()]
X = list(df_bert['inserted_at'])
y = list(df_bert['type'])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(X, truncation=True, padding=True, max_length=512)

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

dataset = NewsDataset(encodings, y_encoded)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

from transformers import DistilBertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_encoder.classes_))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_result = trainer.evaluate()
print(eval_result)

preds_output = trainer.predict(val_dataset)
pred_labels = np.argmax(preds_output.predictions, axis=1)
true_labels = preds_output.label_ids

print(classification_report(
    true_labels,
    pred_labels,
    labels=list(range(len(label_encoder.classes_))),
    target_names=label_encoder.classes_
))
