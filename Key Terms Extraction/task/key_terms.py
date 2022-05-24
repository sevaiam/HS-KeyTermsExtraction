# Write your code here
import string
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import pandas as pd
import itertools
import collections
xml_path = "news.xml"
root = etree.parse(xml_path).getroot()
corpus = root[0]
wnl = WordNetLemmatizer()
stop_word_list = list(string.punctuation) + stopwords.words('english')
tfidf_vectorizer = TfidfVectorizer()

# dataset = [news[1].text for news in corpus]

articles = []
dataset = []
for x, news in enumerate(corpus):
    tokenized_list = word_tokenize(news[1].text.lower())
    lemmatized_pre_stopwords = [wnl.lemmatize(word) for word in tokenized_list]
    clean_list = [word for word in lemmatized_pre_stopwords if word not in stop_word_list]
    noun_list = []
    for word in clean_list:
        if nltk.pos_tag([word])[0][1] == 'NN':
            noun_list.append(word)

    counted = Counter(sorted(noun_list, reverse=True)).most_common(5)
    articles.append([])
    articles[x].append(news[0].text + ':')
    articles[x].append(noun_list)
    dataset.append(' '.join(noun_list))

tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(dataset)
for num, article in enumerate(articles):
    first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[num]
    # df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names_out(), columns=["tfidf"])
    # df.sort_index(ascending=True, inplace=True)
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names_out(), columns=["tfidf"])

    df.sort_values(by=["tfidf"],ascending=False, inplace=True)
    # print(df)
    top_10_unsorted = dict(itertools.islice(df.to_dict('dict')['tfidf'].items(), 15))
    top_10 = sorted(top_10_unsorted.items(), key=lambda x:x[1], reverse=True)
    sorted_dict = defaultdict(list)
    for name, tfidf in top_10:
        sorted_dict[tfidf].append(name)
        sorted_dict[tfidf].sort(reverse=True)
    sorted_values = sorted_dict
    print(article[0])
    # print(sorted_dict.values())
    to_print = []
    for num, value in enumerate(sorted_dict.values()):

        counter = 0
        for i in value:
            to_print.append(i)

    print(' '.join(to_print[:5]))
    # print(' '.join(i for i in sorted_dict.values()))
