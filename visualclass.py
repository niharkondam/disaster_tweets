import streamlit as st
import nltk

#nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import demoji
from spellchecker import SpellChecker
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def rem_links(text):
    ws = []
    for w in text.split():
        if w[0:4] == 'http':
            None
        else:
            ws.append(w)
    text = " ".join(ws)
    return text


def expand_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    text = ' '.join(expanded_words)
    return text


def emoji_text(text):
    emdict = demoji.findall(text)

    for k in emdict.keys():
        text = text.replace(k, ' '+emdict[k].split(':')[0]+' ')
    text = text.replace('  ',' ')
    return text


def rem_sc(text):
    txt = ''

    for c in text:

        if c.isalnum() or c == ' ':
            txt += c
        else:
            txt += ' '
    txt = txt.replace('  ', ' ')
    return txt


def break_compounds(text):
    ws = []
    for w in text.split():
        wn = ''
        for i, c in enumerate(w):

            if c.lower() != c:
                if i != 0:
                    if w[i - 1].lower() == w[i - 1]:
                        wn += ' ' + c
                    else:
                        wn += c
                else:
                    wn += c
            else:
                wn += c
        ws.append(wn)
    text = ' '.join(ws)
    return text


spell = SpellChecker()


def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            cw = spell.correction(word)
            if cw:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)
        else:
            corrected_text.append(word)
    #    print(corrected_text)
    return " ".join(corrected_text)


def lem(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text


vectorizer = pickle.load(open('/Users/niharkondam/Projects/disaster_tweets/vectorizer_model.pkl', 'rb'))
lr_model = pickle.load(open('/Users/niharkondam/Projects/disaster_tweets/lr_model.pkl', 'rb'))

st.header('Disaster Tweet Detection')
text = st.text_input('Type your tweet here:', "yall 🔥🔥🔥 for sunburn this year?")


text = rem_links(text)
st.write(f'removed links:\n {text}')

text = expand_contractions(text)
st.write(f'expanded contractions:\n {text}')

text = emoji_text(text)
st.write(f'replaced emojis with text:\n {text}')

text = rem_sc(text)
st.write(f'removed special characters:\n {text}')

text = break_compounds(text)
st.write(f'broke down coumpounds:\n {text}')

text = text.lower()
st.write(f'lower cased:\n {text}')

text = correct_spellings(text)
st.write(f'corrected spellings:\n {text}')

text = lem(text)
st.write(f'lemmatized and removed stopwords:\n {text}')

if lr_model.predict(vectorizer.transform([text]))[0] == 1:
    st.write('This tweet is about a disaster')
else:
    st.write('This tweet is not about a disaster')
