import pandas as pd
import csv
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as file:  
    model = pickle.load(file)

pkl_tfidf = "tfidf.pk"
with open(pkl_tfidf, 'rb') as file:  
    tfidf = pickle.load(file)

id_to_genre = 'id_to_genre.pk'
with open(id_to_genre, 'rb') as file:  
    id_to_genre = pickle.load(file)


df = pd.read_csv('data/Lyrics-Genre-Test.csv')

features = tfidf.transform(df.Lyrics)
# test first 2 predictions
with open('results.csv',mode='w') as result_file:
    writer = csv.writer(result_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Id','Prediction 1','Prediction 2'])
    values = model.predict_proba(features)
    for idx,row in enumerate(values):
        order = np.argpartition(np.array(values[idx]), -2)[-2:]
        writer.writerow([df.Track_id[idx],id_to_genre[order[0]],id_to_genre[order[1]]])

