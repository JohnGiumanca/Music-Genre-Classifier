import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# prepare dataset
df = pd.read_csv('data/Lyrics-Genre-Train.csv')
col = ['Genre', 'Lyrics']
df = df[col]

# assign id to genre
df['genre_id'] = df['Genre'].factorize()[0]
genre_id_df = df[['Genre', 'genre_id']].drop_duplicates().sort_values('genre_id')
genre_to_id = dict(genre_id_df.values)
id_to_genre = dict(genre_id_df[['genre_id', 'Genre']].values)

# save dict
with open('id_to_genre.pk', 'wb') as fin:
	pickle.dump(id_to_genre, fin)

# equilibrate dataset
# df = df.groupby('Genre').head(1000)

# visualize
# fig = plt.figure(figsize=(8,6))
# df.groupby('Genre').Lyrics.count().plot.bar(ylim=0)
# plt.show()

# bag of words
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Lyrics).toarray()
labels = df.genre_id
with open('tfidf.pk', 'wb') as fin:
	pickle.dump(tfidf, fin)
# print(features.shape)

# train
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
svm = LinearSVC()
model_svm = CalibratedClassifierCV(svm) 
model_svm = model_svm.fit(X_train, y_train)
model_logistic_reg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
model_logistic_reg = model_logistic_reg.fit(X_train, y_train)

#choose model
model = model_svm
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

#save model
filename = 'model.pkl'
with open(filename, 'wb') as file:  
    pickle.dump(model, file)




# y_test.to_csv("testdata.csv")

# test_cases = pd.read_csv("testdata.csv")
# with open('results.csv',mode='w') as result_file:
#     writer = csv.writer(result_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Id','Prediction 1','Prediction 2'])
#     values = model.predict_proba(X_test)
#     for idx,row in enumerate(values):
#         order = np.argpartition(np.array(values[idx]), -2)[-2:]
#         writer.writerow([row[0],model.classes_[order[0]],model.classes_[order[1]]])

# y_pred = pd.read_csv("results.csv")
# print(accuracy_score(y_test, y_pred.iloc[:,1]) + accuracy_score(y_test, y_pred.iloc[:,2]))


