# Libraries
import numpy as np
import pandas as pd

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
import advertools as adv

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFBertModel
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Import data, dataset available on Kaggle 'Natural Language Processing with Disaster Tweets'
test = pd.read_csv('../input/nlp-getting-started/test.csv')
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test.drop('id', axis=1, inplace=True)
train.drop('id', axis=1, inplace=True)

train['location'].dropna(inplace=True)
test['location'].dropna(inplace=True)

train.fillna('', inplace = True)
test.fillna('', inplace = True)

# Cleaning data
def keyclean(text):
    try:
        text = text.split('%20')
        text = ' '.join(text)
        return text
    except:
        return text
    
def cleaning(text):
    clean_text = text.translate(str.maketrans('','',string.punctuation)).lower()
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
    
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))
        
    return ' '.join(sentence)
    
train['keyword'] = train['keyword'].apply(keyclean)
test['keyword'] = test['keyword'].apply(keyclean)

train['text'] = train['text'].apply(cleaning)
test['text'] = test['text'].apply(cleaning)

# Feature engineering
def count_chars(text):
    return len(text)

def count_words(text):
    return len(text.split())

def unique_words(text):
    return len(set(text.split()))

def hashtag_counts(text):
    return adv.extract_hashtags(text)['hashtag_counts'][0]

def mention_counts(text):
    return adv.extract_mentions(text)['mention_counts'][0]

def question_counts(text):
    return adv.extract_questions(text)['question_mark_counts'][0]

def url(text):
    count = 0
    
    text = text.split()
    for i in text:
        if i.startswith('http'):
            count = count+1
    
    return count
  
train['chars'] = train['text'].apply(count_chars)
train['words'] = train['text'].apply(count_words)
train['unique_words'] = train['text'].apply(unique_words)
train['word_length'] = train['chars'] / train['words']
train['hashtag'] = train['text'].apply(hashtag_counts)
train['mention'] = train['text'].apply(mention_counts)
train['question'] = train['text'].apply(question_counts)
train['url'] = train['text'].apply(url)

test['chars'] = test['text'].apply(count_chars)
test['words'] = test['text'].apply(count_words)
test['unique_words'] = test['text'].apply(unique_words)
test['word_length'] = test['chars'] / test['words']
test['hashtag'] = test['text'].apply(hashtag_counts)
test['mention'] = test['text'].apply(mention_counts)
test['question'] = test['text'].apply(question_counts)
test['url'] = test['text'].apply(url)

# Encoding
AUTO = tf.data.experimental.AUTOTUNE
EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 25

def encode(texts, tokenizer, max_len=MAX_LEN):
    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    
    for text in texts:
        token = tokenizer(text, max_length=max_len, 
                          truncation=True, 
                          padding='max_length',
                          add_special_tokens=True)
        input_ids.append(token['input_ids'])
        token_type_ids.append(token['token_type_ids'])
        attention_mask.append(token['attention_mask'])
        
    return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)

# Load BERT
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.save_pretrained('.')

# Apply seperator and merge keyword and text columns
sep = tokenizer.sep_token
train['inputs'] = train.keyword + sep + train.text

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'],
                                                    test_size = 0.2,
                                                    random_state = 52)

X_train = encode(X_train.astype(str), tokenizer)
X_test = encode(X_test.astype(str), tokenizer)

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

# Build model
def build_model(bert_model, max_len=MAX_LEN):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    
    sequence_output = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    
    clf_output = sequence_output[:, 0, :]
    clf_output = Dropout(.1)(clf_output)
    out = Dense(3, activation='softmax')(clf_output)
    
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)
    model.compile(Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

transformer_layer = (TFBertModel.from_pretrained('bert-base-cased'))
model = build_model(transformer_layer, max_len=MAX_LEN)

# Training model
history = model.fit(
    train_dataset,
    steps_per_epoch=200,
    validation_data=valid_dataset,
    epochs=10
)

# Evaluate model
history = pd.DataFrame(history.history)
print('Model Performance:')
print(('Objective: Accuracy: {:0.4f}')\
     .format(history['accuracy'].max()))
print(('Validation Accuracy: {:0.4f}')\
     .format(history['val_accuracy'].max()))
print(('Loss               : {:0.4f}')\
     .format(history['loss'].min()))
print(('Validation Loss    : {:0.4f}')\
     .format(history['val_loss'].min()))


