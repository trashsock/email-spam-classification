import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #since body of email is only from 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)

    return dictionary


def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix


# Creating a dictionary of words with its frequency
train_dir = 'train-mails'
dictionary = make_Dictionary(train_dir)

# Preparing feature vectors per training mail and its labels
train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier
m1 = MultinomialNB()
m2 = LinearSVC()
m1.fit(train_matrix,train_labels)
m2.fit(train_matrix,train_labels)

# Testing the unseen mails for Spam
test_dir = 'test-mails'
test_matrix = extract_features(test_dir)
test_lbls = np.zeros(260)
test_lbls[130:260] = 1
res1 = m1.predict(test_matrix)
res2 = m2.predict(test_matrix)

print confusion_matrix(test_lbls, res1)
print confusion_matrix(test_lbls, res2)