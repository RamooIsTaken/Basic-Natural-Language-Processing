import pandas as pd
import numpy as np
import re
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


df = pd.read_csv('output.csv')
nltk.download('stopwords')
ps = PorterStemmer()
cv = CountVectorizer(max_features=2000)

derlem = []
for i in range(df.shape[0]):
    yorum = re.sub('[^a-zA-Z]',' ',df[' Review'][i])
    yorum = yorum.lower().split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)

X = cv.fit_transform(derlem).toarray()
y = df.iloc[:,1].values

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

gnb = GaussianNB()
gnb.fit(x_train,y_train)
yPred = gnb.predict(x_test)

cm = confusion_matrix(y_test,yPred)
print(cm)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

