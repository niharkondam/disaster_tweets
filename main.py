import streamlit as st
import nltk

nltk.download('all')
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
    return text.strip()


def expand_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    text = ' '.join(expanded_words)
    return text.strip()


def emoji_text(text):
    emdict = demoji.findall(text)

    for k in emdict.keys():
        text = text.replace(k, ' '+emdict[k].split(':')[0]+' ')
    text = text.replace('  ', ' ')
    return text.strip()


def rem_sc(text):
    txt = ''

    for c in text:

        if c.isalnum() or c == ' ':
            txt += c
        else:
            txt += ' '
    txt = txt.replace('  ', ' ')
    return txt.strip()


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
    return text.strip()


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
    return " ".join(corrected_text).strip()


def lem(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text.strip()


vectorizer = pickle.load(open('vectorizer_model.pkl', 'rb'))
lr_model = pickle.load(open('lr_model1.pkl', 'rb'))

st.image('nlp1-cover.jpg')
st.header('Disaster Tweet Detection')
#text = st.text_input('Type your tweet here:', "I can't belive my backyard is on fire🔥😱. https://howtomakeitrain.com #StayinAlive", placeholder = '')
text = st.text_input('Type your tweet here:', placeholder="I can't belive my backyard is on fire🔥😱. https://howtomakeitrain.com #StayinAlive")

if text == '':
    text = "I can't belive my backyard is on fire🔥😱. https://howtomakeitrain.com #StayinAlive"


text1 = rem_links(text)
text2 = expand_contractions(text1)
text3 = emoji_text(text2)
text4 = rem_sc(text3)
text5 = break_compounds(text4)
text6 = text5.lower()
text7 = correct_spellings(text6)
text8 = lem(text7)

if lr_model.predict(vectorizer.transform([text8]))[0] == 1:
    st.write(':red[DISASTER]')
else:
    st.write(':green[Not a Disaster]')

if text1 != text:
    st.markdown("**Remove the urls**")
    st.text(text1)
if text2 != text1:
    st.markdown("**Expand the contractions**")
    st.text(text2)
if text3 != text2:
    st.markdown("**Substitute the emojis**")
    st.text(text3)
if text4 != text3:
    st.markdown("**Remove the Special Characters**")
    st.text(text4)
if text5 != text4:
    st.markdown("**Breakdown Compound Words**")
    st.text(text5)
if text6 != text5:
    st.markdown("**Convert to lower cases**")
    st.text(text6)
if text7 != text6:
    st.markdown("**Correct the spellings**")
    st.text(text7)
if text8 != text7:
    st.markdown("**Remove the stopwords and lemmatize**")
    st.text(text8)
