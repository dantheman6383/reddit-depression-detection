import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import string
!pip install happiestfuntokenizing
from happiestfuntokenizing.happiestfuntokenizing import Tokenizer

from google.colab import drive
drive.mount('/content/drive')

FILEPATH = '/content/drive/MyDrive/student.pkl'

depression_subreddits = ["Anger",
    "anhedonia", "DeadBedrooms",
    "Anxiety", "AnxietyDepression", "HealthAnxiety", "PanicAttack",
    "DecisionMaking", "shouldi",
    "bingeeating", "BingeEatingDisorder", "EatingDisorders", "eating_disorders", "EDAnonymous",
    "chronicfatigue", "Fatigue",
    "ForeverAlone", "lonely",
    "cry", "grief", "sad", "Sadness",
    "AvPD", "SelfHate", "selfhelp", "socialanxiety", "whatsbotheringyou",
    "insomnia", "sleep",
    "cfs", "ChronicPain", "Constipation", "EssentialTremor", "headaches", "ibs", "tinnitus",
    "AdultSelfHarm", "selfharm", "SuicideWatch",
    "Guilt", "Pessimism", "selfhelp", "whatsbotheringyou"
]

def load():
  """Load pickles"""
  # open pickle file
  with open(FILEPATH, 'rb') as f:
    # load pickle
    data = pd.read_pickle(f)
  return data

data = load()

def dataset_generation():
  """Build control and symptom datasets"""
  datasets = []
  # use dict to associate each symptom to the subreddits that comprise them
  symptom_to_sub = {}

  # get one dataset for each symptom using subreddit column value
  anger = data.loc[data['subreddit'] == "Anger"]
  datasets.append(anger)
  symptom_to_sub["Anger"] = ["Anger"]

  anhedonia = data.loc[(data['subreddit'] == "anhedonia") | (data['subreddit'] == "DeadBedrooms")]
  datasets.append(anger)
  symptom_to_sub["Anhedonia"] = ["anhedonia", "DeadBedrooms"]

  anxiety = data.loc[(data['subreddit'] == "Anxiety") | (data['subreddit'] == "AnxietyDepression") | (data['subreddit'] == "HealthAnxiety") | (data['subreddit'] == "PanicAttack")]
  datasets.append(anxiety)
  symptom_to_sub["Anxiety"] = ["Anxiety", "AnxietyDepression", "HealthAnxiety", "PanicAttack"]

  # excluded symptoms should still be in depression dataset, but should not get their own symptom dataset
  concentration_deficit = data.loc[(data['subreddit'] == "DecisionMaking") | (data['subreddit'] == "shouldi")]

  disordered_eating = data.loc[(data['subreddit'] == "bingeeating") | (data['subreddit'] == "BingeEatingDisorder") | (data['subreddit'] == "EatingDisorders") | (data['subreddit'] == "eating_disorders") | (data['subreddit'] == "EDAnonymous")]
  datasets.append(anxiety)
  symptom_to_sub["Disordered eating"] = ["bingeeating", "BingeEatingDisorder", "EatingDisorders", "eating_disorders", "EDAnonymous"]

  fatigue = data.loc[(data['subreddit'] == "chronicfatigue") | (data['subreddit'] == "Fatigue")]

  loneliness = data.loc[(data['subreddit'] == "ForeverAlone") | (data['subreddit'] == "lonely")]
  datasets.append(anxiety)
  symptom_to_sub["Loneliness"] = ["ForeverAlone", "lonely"]

  sad_mood = data.loc[(data['subreddit'] == "cry") | (data['subreddit'] == "grief") | (data['subreddit'] == "sad") | (data['subreddit'] == "Sadness")]
  datasets.append(sad_mood)
  symptom_to_sub["Sad mood"] = ["cry", "grief", "sad", "Sadness"]

  self_loathing = data.loc[(data['subreddit'] == "AvPD") | (data['subreddit'] == "SelfHate") | (data['subreddit'] == "selfhelp") | (data['subreddit'] == "socialanxiety") | (data['subreddit'] == "whatsbotheringyou")]
  datasets.append(self_loathing)
  symptom_to_sub["Self-loathing"] = ["AvPD", "SelfHate", "selfhelp", "socialanxiety", "whatsbotheringyou"]

  sleep_problem = data.loc[(data['subreddit'] == "insomnia") | (data['subreddit'] == "sleep")]
  datasets.append(sleep_problem)
  symptom_to_sub["Sleep problem"] = ["insomnia", "sleep"]

  somatic_complaint = data.loc[(data['subreddit'] == "cfs") | (data['subreddit'] == "ChronicPain") | (data['subreddit'] == "Constipation") | (data['subreddit'] == "EssentialTremor") | (data['subreddit'] == "headaches") | (data['subreddit'] == "ibs") | (data['subreddit'] == "tinnitus")]
  datasets.append(somatic_complaint)
  symptom_to_sub["Somatic complaint"] = ["cfs", "ChronicPain", "Constipation", "EssentialTremor", "headaches", "ibs", "tinnitus"]

  suicide = data.loc[(data['subreddit'] == "AdultSelfHarm") | (data['subreddit'] == "selfharm") | (data['subreddit'] == "SuicideWatch")]

  worthlessness = data.loc[(data['subreddit'] == "Guilt") | (data['subreddit'] == "Pessimism") | (data['subreddit'] == "selfhelp") | (data['subreddit'] == "whatsbotheringyou")]
  datasets.append(worthlessness)
  symptom_to_sub["Worthlessness"] = ["Guilt", "Pessimism", "selfhelp", "whatsbotheringyou"]

  # depression dataset
  datasets.insert(0, pd.concat([anger, anhedonia, anxiety, disordered_eating, loneliness, sad_mood, self_loathing, sleep_problem, somatic_complaint, worthlessness, concentration_deficit, suicide, fatigue]))

  # map each author to index depression posting
  author_to_index = {}
  for _, post in datasets[0].iterrows():
    # if we've seen the author before, update their index post if the current post is earlier than the post already stored
    if post['author'] in author_to_index:
      author_to_index[post['author']] = min(post['created_utc'], author_to_index[post['author']])
    # haven't seen this author; store current post as their index post thus far
    else:
      author_to_index[post['author']] = post['created_utc']

  # Filter non-depression posts to include only those older than 180 days from the author's earliest depression posting
  c = data.loc[~data['subreddit'].isin(depression_subreddits)]
  c = c.loc[(c['author'].isin(author_to_index.keys())) & (c['author'].map(author_to_index) - c['created_utc'] >= 15552000)]

  datasets.insert(0, c)
  return datasets, symptom_to_sub

datasets, symptom_to_sub = dataset_generation()

def tokenize():
  dataset_tokens = []
  tokenizer = Tokenizer()

  for dataset in datasets:
    # take the 'text' column, remove punctuation, and then tokenize (which automatically lowercases), and cast to list
    dataset_tokens.append(list(dataset['text'].apply(lambda x: tokenizer.tokenize(x.translate(str.maketrans('', '', string.punctuation))))))
  return dataset_tokens

dataset_tokens = tokenize()

from collections import Counter
def stop_words():
  control = dataset_tokens[0]

  # use Counter to get token frequencies
  counter = Counter([token for row in control for token in row])

  # get top 100 words
  stop_words = counter.most_common(100)
  return stop_words

stop_words = stop_words()
# filter stop words from control and depression tokens
dataset_tokens[0] = [[token for token in row if token not in stop_words] for row in dataset_tokens[0]]
dataset_tokens[1] = [[token for token in row if token not in stop_words] for row in dataset_tokens[1]]

from gensim.models import LdaMulticore
from gensim import corpora

=vocab = set()
corpus = dataset_tokens[0] + dataset_tokens[1]

# go through control + depression and build vocab
for post in corpus:
  for token in post:
    vocab.add(token)

# map id to word for use in LdaMulticore
counter = Counter(token for post in corpus for token in post)
temp = [x[0] for x in counter.most_common(len(vocab))]
idx2word = dict(enumerate(temp))

# use corpora.Dictionary to make a mapping based off corpus that can then be turned into BOW for LdaMulticore
dictionary = corpora.Dictionary(corpus)
corpus = [dictionary.doc2bow(post) for post in corpus]

# train LdaMulticore on control + depression posts
lda = LdaMulticore(corpus, num_topics=200, id2word=idx2word, minimum_probability=0.05)

# get topic distribution for each post in control + depression set
topics = [lda.get_document_topics(post) for post in corpus]
# example of top 10 words associated with first topic
lda.show_topic(0, topn=10)

# build feature matrix
X = np.zeros((len(corpus), 200))

for i in range(len(topics)):
  for topic, prob in topics[i]:
    # each matrix element is the probability of that topic for the post
    X[i, topic] = prob

# build Y (labels matrix)
def build_labels():
  labels = []

  for symptom in ["Anger", "Anhedonia", "Anxiety", "Disordered eating", "Loneliness", "Sad mood", "Self-loathing", "Sleep problem", "Somatic complaint", "Worthlessness"]:
    # store labels for a symptom in 1-d matrix
    Y = np.zeros(len(corpus))
    i = 0

    for _, row in (pd.concat([datasets[0], datasets[1]])).iterrows():
      # if the post's subreddit is associated with the symptom, label it 1, otherwise the label stays 0
      if row.subreddit in symptom_to_sub[symptom]:
        Y[i] = 1
      i = i + 1
    labels.append(Y)
  return labels
Y = build_labels()

from transformers import RobertaModel, RobertaTokenizer
import torch

tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
model = RobertaModel.from_pretrained('distilroberta-base')

model.to("cuda")
model.eval()

# get the text from each post in control + depression dataset for tokenizing
d = pd.concat([datasets[0], datasets[1]])
text = [post['text'] for _, post in d.iterrows()]

embeddings = []
for post in text:
  # tokenize post text
  inputs = tokenizer(post, return_tensors="pt", truncation=True, padding=True, max_length=512)
  inputs.to("cuda")

  with torch.no_grad():
    # use ** to automatically assign input_ids and attention_mask from BatchEncoding from tokenizer; also get hidden states for next line
    outputs = model(**inputs, output_hidden_states=True)

  # get hidden representation from 5th layer according to handout
  hidden_states = outputs.hidden_states[5]

  embedding = hidden_states.squeeze(0)
  avg_embedding = embedding.mean(dim=0)

  # store embedding on cpu and convert to numpy array
  embeddings.append(avg_embedding.cpu().numpy())

embeddings = np.array(embeddings)

def main(X, y):
  """
  Here's the basic structure of the main block! It should run
  5-fold cross validation with random forest to evaluate your RoBERTa and LDA
  performance.
  """
  # evaluate each symptom vs control
  for symptom in y:
    # classifier and kfold work together to cross_validate the feature matrix X based on the symptom labels with AUC scoring
    rf_classifier = RandomForestClassifier(n_estimators = 85, max_depth=15, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True)
    results = cross_validate(rf_classifier, X=X, y=symptom, cv=cv, scoring='roc_auc', return_train_score=True)

    print("LDA")
    print("Train", results['train_score'])
    print("Test", results['test_score'])

    # do same thing but with embeddings instead of feature matrix
    rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True)
    results = cross_validate(rf_classifier, X=embeddings, y=symptom, cv=cv, scoring='roc_auc', return_train_score=True)

    print("Roberta")
    print("Train", results['train_score'])
    print("Test", results['test_score'])

main(X, Y)
