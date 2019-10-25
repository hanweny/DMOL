import pickle
import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from gensim.utils import simple_preprocess

STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 

def save_object(obj, path):
    pickle.dump(obj, open(path, 'wb'))
    
def load_object(path):
    return pickle.load(open(path, 'rb'))

def nlp_preprocess(text, use_stemmer = False):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            if not use_stemmer:
                result.append(lemmatizer.lemmatize(token, pos='v'))
            else:
                result.append(stemmer.stem(lemmatizer.lemmatize(token, pos='v')))
    return " ".join(result)

def generate_wordcloud(text):
    text_to_generate = text
    if type(text) == list:
        text_to_generate = " ".join([" ".join(i) for i in text]) if type(text[0]) == list else " ".join(text)
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = 'white', collocations = False).generate(text_to_generate)
    plt.figure(figsize = (13.8, 6))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()
    
    