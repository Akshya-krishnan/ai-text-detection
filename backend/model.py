import pickle
from collections import Counter
import numpy as np
import pandas as pd

# NLP
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Load models
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# ---------------- FEATURE ENGINEERING ---------------- #

def add_features(data):

    punctuation_types = ['.', ',', "'", '"', '-', '?', ':', ')', '(', '!', '/', ';', '_', ']', '[']

    data = data.copy()

    def count_punctuation(text):
        return (sum(char in punctuation_types for char in text) / len(text)) * 100 if len(text) > 0 else 0

    def richness(text):
        tokens = word_tokenize(text)
        return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

    def count_char_percent(text, char):
        return (text.count(char) / len(text)) * 100 if len(text) > 0 else 0

    def get_sentiment(text):
        return sia.polarity_scores(text)

    def expand_dict_column(data, column):
        df = pd.DataFrame(data[column].tolist())
        data = pd.concat([data, df], axis=1)
        return data.drop(columns=[column])

    def get_pos_tags(text):
        pos_tags = nltk.pos_tag(word_tokenize(text))
        counts = Counter(tag for word, tag in pos_tags)
        total = len(pos_tags)

        if total == 0:
            return {k: 0 for k in ["Nouns","Pronouns","Verbs","Adjectives","Adverbs","Prepositions","Conjunctions","Interjections"]}

        return {
            "Nouns": (counts["NN"] + counts["NNS"]) / total,
            "Pronouns": (counts["PRP"] + counts["PRP$"]) / total,
            "Verbs": (counts["VB"] + counts["VBD"] + counts["VBG"] + counts["VBN"] + counts["VBP"] + counts["VBZ"]) / total,
            "Adjectives": (counts["JJ"] + counts["JJR"] + counts["JJS"]) / total,
            "Adverbs": (counts["RB"] + counts["RBR"] + counts["RBS"]) / total,
            "Prepositions": counts["IN"] / total,
            "Conjunctions": counts["CC"] / total,
            "Interjections": counts["UH"] / total,
        }

    def entity_ratio(text):
        ents = nlp(text).ents
        return len(ents) / len(text) if len(text) > 0 else 0

    def stopword_percent(text):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = word_tokenize(text)
        return (sum(1 for w in words if w in stop_words) / len(words)) * 100 if len(words) > 0 else 0

    def uppercase_percent(text):
        return sum(1 for c in text if c.isupper()) / len(text) * 100 if len(text) > 0 else 0

    def flesch_reading_ease(text):
        words = text.split()
        sentences = text.count('.') + text.count('?') + text.count('!')
        syllables = sum(sum(1 for c in word if c.lower() in 'aeiou') for word in words)
        if sentences == 0: sentences = 1
        return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words)) if words else 0

    def gunning_fog(text):
        words = text.split()
        sentences = text.count('.') + text.count('?') + text.count('!')
        complex_words = sum(1 for word in words if len(word) > 7)
        if sentences == 0: sentences = 1
        return 0.4 * ((len(words)/sentences) + 100*(complex_words/len(words))) if words else 0

    def avg_sentence_length(text):
        sentences = sent_tokenize(text)
        return sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    # FEATURES
    data['punctuation%'] = data['text'].apply(count_punctuation)
    data['Richness'] = data['text'].apply(richness)

    data['%_comma'] = data['text'].apply(lambda x: count_char_percent(x, ','))
    data['%_period'] = data['text'].apply(lambda x: count_char_percent(x, '.'))
    data['%_q_mark'] = data['text'].apply(lambda x: count_char_percent(x, '?'))

    # SENTIMENT (VERY IMPORTANT)
    data['sentiment'] = data['text'].apply(get_sentiment)
    data = expand_dict_column(data, 'sentiment')

    # POS
    data['POS_tags'] = data['text'].apply(get_pos_tags)
    data = expand_dict_column(data, 'POS_tags')

    data['Entities_ratio'] = data['text'].apply(entity_ratio)
    data['%_stopwords'] = data['text'].apply(stopword_percent)
    data['%_uppercase'] = data['text'].apply(uppercase_percent)

    data['text_length'] = data['text'].apply(len)
    data['num_words'] = data['text'].apply(lambda x: len(x.split()))
    data['AverageSentenceLength'] = data['text'].apply(avg_sentence_length)

    data['FleschReadingEase'] = data['text'].apply(flesch_reading_ease)
    data['GunningFogIndex'] = data['text'].apply(gunning_fog)

    return data


# ---------------- PREDICTION ---------------- #

def predict_generated(text_input):

    df = pd.DataFrame({'text': [text_input]})

    df = add_features(df)
    df = df.fillna(0)

    # Load models ONCE
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    doc2vec = pickle.load(open('models/doc2vec_model.pkl', 'rb'))
    svm = pickle.load(open('models/svm_model.pkl', 'rb'))

    # SCALE
    text_col = df['text']
    features = df.drop(columns=['text'])

    expected_scaler_features = scaler.feature_names_in_

    for col in expected_scaler_features:
        if col not in features.columns:
            features[col] = 0

    features = features[expected_scaler_features]

    scaled = scaler.transform(features)
    scaled = pd.DataFrame(scaled, columns=expected_scaler_features)

    df = pd.concat([text_col, scaled], axis=1)

    # DOC2VEC FIX
    tokens = word_tokenize(df['text'].iloc[0].lower())
    vector = doc2vec.infer_vector(tokens)

    for i in range(len(vector)):
        df[f'doc2vec_{i}'] = vector[i]

    df.drop(columns=['text'], inplace=True)

    # MATCH SVM FEATURES
    expected = svm.feature_names_in_

    for col in expected:
        if col not in df.columns:
            df[col] = 0

    df = df[expected]

    # PREDICT
    prob = svm.predict_proba(df)[0][1]

    print("DEBUG Probability:", prob)  # 🔥 important

    # BETTER THRESHOLD
    if prob > 0.5:
        return "AI Generated"
    else:
        return "Human Written"