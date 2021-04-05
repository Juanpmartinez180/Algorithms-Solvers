#Natural languaje processing algorithm

#Library import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset import
""".tsv format means 'tab separated values'
In NLP, comma separator is not the best option. Beacause, it's normal to 
find a comma in a text.
Other solution is using "" to separate text in the dataset, but it need a data 
revision to make sure that the separator is not used for others purposes.

note = In read_csv function, we can use arguments to set a delimiter and also
to ignore delimiters like "". In this case we use option 3 for this purpose.

stopwords are words that doesn't mean anything to the algorithm nor doesn't
change the idea of the text """

dataset = pd.read_csv( 'Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3 )
column = np.size(dataset, 0)

#Text cleaning
""" Clean all the strange characters from the text, and replace them with a white space
    Only keep letters in lower-upper case
    In the second step we transform the text to a lower case (minúsculas)
    In the third we split the string text in to a list of words
    Finally we clean the list word by word keeping only the words that isn't in 
    the stopwords library imported.
    The stem function is also executed in the last instance of cleaning. This function
    only keep the root of the words processed, it means that for a different words with
    similar meaning the stemmer just keep the root of the words, the escencial word
    In a last step we combine all the words again in to a string characters and add it
    to a corpus list to ingest the NLP algortihm"""

import re      #Regular expression library
import nltk    #Natural languaje tool kit library

nltk.download('stopwords')      #Download stopwords library (all languajes)
from nltk.corpus import stopwords #Import the stopwords downloaded to the cleaning algorithm
from nltk.stem.porter import PorterStemmer

corpus = []     #List to save the cleaned test
for i in range (0, column):
    review = re.sub( '[^a-zA-Z]',' ' ,dataset['Review'][i] ) 
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words('english')) ] 
    review = ' '.join(review)
    corpus.append(review)                 

#Bag of words 
"""The idea is transform the words to vectors. This process is called tokenization
 This implementation produces a sparse representation of the counts
 A spare matrix represent how often a word is reapeated in a given dataset
 max_features only keep the most frequent words 
 Once bag of words is finished, we can apply any classification/regression algorithm
 to create a prediction using training and testing datasets"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#Dataset split in to a training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fit the classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Result prediction using test set
y_pred = classifier.predict(X_test)

#Results validation using confussion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)








































