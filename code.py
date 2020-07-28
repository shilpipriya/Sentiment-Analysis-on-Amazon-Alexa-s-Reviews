#As many of us might share same doubt: Are these voice assistants really improving our lives?
#Lots of people think they will only make us lazier. 
#While some of us think they can have more time for works or other 
#things with the help of voice assistants. I'll analyze these Alexa 
#reviews to see if the users really think Alexa make their lives 
#better.

# Importing the libraries
# for basic operations
import numpy as np
import pandas as pd

# for basic visualizations
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Importing the dataset
dataset = pd.read_csv('data.tsv',sep='\t')

#prints the length of the dataset
print(len(dataset))

#it prints top 5 rows of the dataset
dataset.head()  

#describes the dataset 
dataset.describe()

#describes the dataset by making groups as per rating 
dataset.groupby('rating').describe()

# checking if there is any null data or not
dataset.isnull().any().any()


#Making a new column to detect how long the text messages are:
dataset['length'] = dataset['verified_reviews'].apply(len)
dataset.head()

#graph to represent the frequency of data with the given length in the dataset 
dataset['length'].plot(bins=50, kind='hist')

#desribes the length column of the dataset 
dataset.length.describe()

#Wow! 2851 characters long review, let's use masking to find this message
dataset[dataset['length'] == 2851]['verified_reviews'].iloc[0]

dataset[dataset['length'] == 1]['verified_reviews'].iloc[0]

#we can infer that most of the Ratings are good for alexa. 
#Around 72.6% people have given Alexa 5 Star rating, 
#which is very good. 14.4% people have given Alexa a 4 Star Rating, 
#which is also good. that means 72.6+14.4 = 87% people have given 
#alexa good rating.
#4.38% people have given alexa an average rating of 3 stars. 
#3.05% people did not like alexa and chose to give only 2 star 
#ratings to alexa whereas 5.11% people hated alexa and decided to 
#give alexa only 1 Star Rating. This a total of 3.05+5.11 = 8.16% 
#people did not like alexa.

ratings = dataset['rating'].value_counts()

#popularity of variation of Amazon Alexa according to reviews of dataset
color = plt.cm.copper(np.linspace(0, 1, 15))
dataset['variation'].value_counts().plot.bar(color = color, figsize = (15, 9))
plt.title('Distribution of Variations in Alexa', fontsize = 20)
plt.xlabel('variations')
plt.ylabel('count')
plt.show()

#Distribution of feedback for Amazon Alexa which says that around 
#92% people gave a positive feedback to Amazon Alexa and only 8% 
#people gave negative feedback to Amazon Alexa. 
#This Suggests that Amazon Alexa is a popular product amongst so 
#many people and only few people did not like it for some 
#unforeseeable factors.

feedbacks = dataset['feedback'].value_counts()
print("Positive feedback: ",(feedbacks[1]/feedbacks.sum())*100,"%")
print("Negative feedback: ",(feedbacks[0]/feedbacks.sum())*100,"%")

#The above Bivariate plot, which plots Variation and ratings to 
#check which of the Variation of Amazon Alexa has been perfoeming 
#best in terms of ratings.
#Walnut finish and Oak Finish have very high ratings, 
#the ratings rangee from 4.5 to 5 which is really impressive, 
#These variation are rare and have high reviews.
#White and Black Variations for Amazon Alexa have low ratings also, 
#as it is the most common variation available for the product that 
#is the reason, why it has ratings varying from 0 to 5.

plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

sns.boxenplot(dataset['variation'], dataset['rating'], palette = 'spring')
plt.title("Variation vs Ratings")
plt.xticks(rotation = 90)
plt.show()

#The above Bivariate plot shows a plot between Rating and Length, 
#We would like to that how much a user is gonna write if he/she is 
#going to give a low rating or a high rating to the product.
#We can see that most of the people who gave 5 star rating to 
#Alexa wrote a very small review in comparison to the people who 
#did not give alexa a 5 star rating. But, the longest reviews are 
#written for the 5 star ratings only. The people who gave alexa, 
#a low rating such as 1 or 2 star rating did not consider writing a 
#longer review maybe because they do not like the product.
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 7)
plt.style.use('fivethirtyeight')

sns.boxplot(dataset['rating'], dataset['length'], palette = 'Blues')
plt.title("Length vs Ratings")
plt.show()

#if review length is a distinguishing feature between positive and negative review:
dataset.hist(column='length', by='feedback', bins=50,figsize=(10,4))

#importing the dataset again
data=pd.read_csv('data.tsv', delimiter = '\t', quoting = 3)

#cleaning the texts and stemming the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]', ' ', data['verified_reviews'][i] )
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
# creating the Bag of words Model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=data.iloc[:,4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#make a list of accuracies to store all the accuracy for different algorithm
accuracies={}
precision={}
recall={}
f_score={}

#MODEL-1
# Fitting Random Forest classifier with 100 trees to the Training set
X_train1=X_train
y_train1=y_train
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train1, y_train1)

X_test1=X_test
y_pred1 = classifier.predict(X_test1)

print("Training Accuracy :", classifier.score(X_train1, y_train1))
print("Testing Accuracy :", classifier.score(X_test1, y_test))

accuracies['Random Forest']=classifier.score(X_test1, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)

#TN FP
#FN TP

accuracy1=(18+576)/(18+36+576)
print(accuracy1)

precision1=576/(36+576)
precision['Random Forest']=precision1
print(precision1)

recall1=576/(0+576)
recall['Random Forest']=recall1
print(recall1)

f_score1=(2*precision1*recall1)/(precision1+recall1)
f_score['Random Forest']=f_score1
print(f_score1)

#FITTING LOGISTIC REGRESSION CLASSIFIER TO OUR TRAINING SET
X_train2=X_train
y_train2=y_train
X_test2=X_test

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train2, y_train2)

# Predicting the Test set results
y_pred2 = classifier_lr.predict(X_test2)

print("Training Accuracy :", classifier_lr.score(X_train2, y_train2))
print("Testing Accuracy :", classifier_lr.score(X_test2, y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#TN FP
#FN TP

accuracy2=(545+34)/(630)
print(accuracy2)
accuracies['Logistic Regression']=accuracy2

precision2=545/(20+545)
print(precision2)
precision['Logistic Regression']=precision2

recall2=545/(31+545)
print(recall2)
recall['Logistic Regression']=recall2

f_score2=(2*precision2*recall2)/(precision2+recall2)
print(f_score2)
f_score['Logistic Regression']=f_score2

#FITTING K_NEAREST_NEIGHBORS MODEL TO OUR TRAINING SET
X_train3=X_train
y_train3=y_train
X_test3=X_test

from sklearn.neighbors import KNeighborsClassifier
# try to find best k value
scoreList = []
for i in range(1,20):
    knn3 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn3.fit(X_train3, y_train3)
    scoreList.append(knn3.score(X_test3, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

# Training the K-NN model on the Training set
cknn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
cknn.fit(X_train3, y_train3)

# Predicting the Test set results
y_pred3 = cknn.predict(X_test3)

print("Training Accuracy :", cknn.score(X_train3, y_train3))
print("Testing Accuracy :", cknn.score(X_test3, y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)

#TN FP
#FN TP

accuracy3=(576+4)/(630)
print(accuracy3)
accuracies['k_nearest_neighbors']=accuracy3

precision3=576/(50+576)
print(precision3)
precision['k_nearest_neighbors']=precision3

recall3=576/(0+576)
print(recall3)
recall['k_nearest_neighbors']=recall3

f_score3=(2*precision3*recall3)/(precision3+recall3)
print(f_score3)
f_score['k_nearest_neighbors']=f_score3

#FITTING SUPPORT VECTOR MACHINE CLASSIFIER TO OUR TRAINING SET 

X_train4=X_train
y_train4=y_train
X_test4=X_test

# Training the SVM model on the Training set
from sklearn.svm import SVC
csvm = SVC(kernel = 'linear', random_state = 0)
csvm.fit(X_train4, y_train4)

# Predicting the Test set results
y_pred4 = csvm.predict(X_test4)

print("Training Accuracy :", csvm.score(X_train4, y_train4))
print("Testing Accuracy :", csvm.score(X_test4, y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)
print(cm4)

#TN FP
#FN TP

accuracy4=(561+29)/(630)
print(accuracy4)
accuracies['SVM']=accuracy4

precision4=561/(25+561)
print(precision4)
precision['SVM']=precision4

recall4=561/(15+561)
print(recall4)
recall['SVM']=recall4

f_score4=(2*precision4*recall4)/(precision4+recall4)
print(f_score4)
f_score['SVM']=f_score4

#FITTING KERNEL SVM TO OUR TRAINING SET 
X_train5=X_train
y_train5=y_train
X_test5=X_test

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
cksvm = SVC(kernel = 'rbf', random_state = 0)
cksvm.fit(X_train5, y_train5)

# Predicting the Test set results
y_pred5 = cksvm.predict(X_test5)

print("Training Accuracy :", cksvm.score(X_train5, y_train5))
print("Testing Accuracy :", cksvm.score(X_test5, y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred5)
print(cm5)

#TN FP
#FN TP

accuracy5=(576+8)/(630)
print(accuracy5)
accuracies['Kernel SVM']=accuracy5

precision5=576/(46+576)
print(precision5)
precision['Kernel SVM']=precision5

recall5=576/(0+576)
print(recall5)
recall['Kernel SVM']=recall5

f_score5=(2*precision5*recall5)/(precision5+recall5)
print(f_score5)
f_score['Kernel SVM']=f_score5

#FITTING NAIVE BAYES TO TRAINING SET 

X_train6=X_train
y_train6=y_train
X_test6=X_test

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
cnb = GaussianNB()
cnb.fit(X_train6, y_train6)

# Predicting the Test set results
y_pred6 = cnb.predict(X_test6)

print("Training Accuracy :", cnb.score(X_train6, y_train6))
print("Testing Accuracy :", cnb.score(X_test6, y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred6)
print(cm6)

#TN FP
#FN TP

accuracy6=(302+30)/(630)
print(accuracy6)
accuracies['Naive Bayes']=accuracy6

precision6=302/(24+302)
print(precision6)
precision['Naive Bayes']=precision6

recall6=302/(274+302)
print(recall6)
recall['Naive Bayes']=recall6

f_score6=(2*precision6*recall6)/(precision6+recall6)
print(f_score6)
f_score['Naive Bayes']=f_score6

#FITTING DECISION TREE CLASSIFIER 

X_train7=X_train
y_train7=y_train
X_test7=X_test

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
cdt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
cdt.fit(X_train7, y_train7)

# Predicting the Test set results
y_pred7 = cdt.predict(X_test7)

print("Training Accuracy :", cdt.score(X_train7, y_train7))
print("Testing Accuracy :", cdt.score(X_test7, y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm7 = confusion_matrix(y_test, y_pred7)
print(cm7)

#TN FP
#FN TP

accuracy7=(566+28)/(630)
print(accuracy7)
accuracies['decision tree']=accuracy7

precision7=566/(26+566)
print(precision7)
precision['decision tree']=precision7

recall7=566/(10+566)
print(recall7)
recall['decision tree']=recall7

f_score7=(2*precision7*recall7)/(precision7+recall7)
print(f_score7)
f_score['decision tree']=f_score7

#Comparing models

#comparing accuracy of different models

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,6))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

#comparing precision of different models

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,6))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Precision %")
plt.xlabel("Algorithms")
sns.barplot(x=list(precision.keys()), y=list(precision.values()), palette=colors)
plt.show()

#comparing recall of different models
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,6))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Recall %")
plt.xlabel("Algorithms")
sns.barplot(x=list(recall.keys()), y=list(recall.values()), palette=colors)
plt.show()

#comparing f_score of different models
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,6))
plt.yticks(np.arange(0,100,10))
plt.ylabel("f_score %")
plt.xlabel("Algorithms")
sns.barplot(x=list(f_score.keys()), y=list(f_score.values()), palette=colors)
plt.show()

#comparing confusion matrix of different models

print("Random Forest\n",cm1)
print("Logistic Regression\n",cm2)
print("KNN\n",cm3)
print("SVM\n",cm4)
print("kSVM\n",cm5)
print("Naive Bayes",cm6)
print("Decision tree\n",cm7)

#RESULT

#Kernel SVM rocks in terms of accuracy!!!
#random forest rocks in terms of f_score!!!

