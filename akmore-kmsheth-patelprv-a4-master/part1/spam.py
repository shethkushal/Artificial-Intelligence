"""
Problem formulation:
Note: Please keep 'ignorewordlist.txt' file in the same directory as the script, this file contains stop words which
would be ignored by the classifier.

Stop words are taken from following link:
"http://www.lextek.com/manuals/onix/stopwords1.html"

Brief Description:

HOW TO RUN:

Training on Naive Bayes:
python spam.py train bayes DOCUMENT_FOLDER model-file
Testing on Naive Bayes:
python spam.py test bayes DOCUMENT_FOLDER model-file

Training on Decision Tree:
python spam.py train dt DOCUMENT_FOLDER model-file
Testing on Decision Tree:
python spam.py test dt DOCUMENT_FOLDER model-file

NAIVE BAYES CLASSIFIER
Training:
not_spam_dict_binary - To store the count of spam words in binary format
spam_dict_binary- To store the count of non spam words in binary format
not_spam_dict - To store the count of spam words in continuous frequency
spam_dict- To store the count of non spam words in continuous frequency

Binary Features:
For each document in the training set, we assign a value of 1 if the word is present in the spam/ non spam document. The count is stored in the respective dictionary
If the word is not present in the document a value of 0 is assigned

Continuous Features:
Assign each word a frequency (number of times) it is found in a document

Pickle is used to create a model file.
The priors- Probability(Document being Spam) and Probability(Document not being spam) is calculated

On running the train mode on Naive Bayes, the top 10 words most associated with spam words will be displayed
Also a list of words with least association with spam words will be displayed.
This will be for both Binary Features and Continous Features.

Training on Naive Bayes is very fast and gets done within 10-15 seconds (On given training set)

Testing:
The Model file which consists of four dictionaries is sent to the test set from the training set.

We clasify each document as Spam or Not spam by using the Naive Bayes approach. The probability of the word being spam or not spam is
multiplied with the prior probability.
If the probability of the document being spam is more than the probability of the document being non spam, we increment the respective spam
or non spam counter.

The same process is then followed for Continuous Features.

A confusion matrix is then created to store the actual and predicted classification done by the system. This results along with confusion matrix and accuracy results
will be displayed on running testing mode on Naive Bayes

Confusion Matrix for Continous Features**
           SPAM       NOT SPAM
SPAM       1172.0     13.0
NOT SPAM   15.0       1354.0
----
Total Accuracy for Classifier(Continous Features):  98.9036805012
----
Confusion Matrix for Binary Features**
           SPAM       NOT SPAM
SPAM       1185.0     0.0
NOT SPAM   7.0        1362.0
----
Total Accuracy for Classifier(Binary Features):  99.7259201253
----

Testing on Naive Bayes is very fast and takes about 10-15 seconds with high accuracy (For given test data)

DECISION TREE CLASSIFIER


Training:
While training we create a matrix in form of list of lists.

The list contains the training documents in form of list. This list contains 0's and 1's in case of binary feature and frequency of words in case of continous features.
This matrix will be used while creating the decision tree.

While creating the list of unique feature(words), we ignore the stop words that are given in "ignorewordlist.txt". This words won't be considered while making tree.
Also those words, that occur in less than 10 documents are removed from the list.

For creating a decision tree,  we iterate over all the features of matrix, in our case words, and on basis of Entropy and Information Gain we are making the splits in the matrix.
This will create a tree recursively with a very simple algorithm. At the end we have a Decision Tree ready to be used for classification.

We have used Pickle library to create the model file. In this model file, we have stored the decision tree that will be used for testing the files.
On running training mode on decision tree, the first four level of tree will be displayed in level order form.

Training on decision takes time according to the given data.
For current training data, it will take around 10-15 minutes to build tree
Within this time, Matrices for both Binary and Continous Features are created and also the two different trees for Binary and Continous Features are created

Testing:
While testing we are reading individual files from spam and nonspam folders and classify it on basis of the decision tree.
We have a count of spam and nonspam that will store the count of documents that are classified as spam and nonspam.

This variables will be used for calculating the accuracy of the classification.

The console will display all the results and the confusion matrix along with the accuracy obtained.
This will be accessed in testing mode.

----
Confusion Matrix for Continous Features**
           SPAM       NOT SPAM
SPAM       1149       36
NOT SPAM   86         1283
----
Total Accuracy for Classifier(Continous Features):  95.2231793265
----
Confusion Matrix for Binary Features**
           SPAM       NOT SPAM
SPAM       1114       71
NOT SPAM   685        684
----
Total Accuracy for Classifier(Binary Features):  70.3993735317
----

Testing for decision tree will be done within 3-4 minutes. THe accuracy is quite decent but low comapred to Naive Bayes Classifier


WHICH ONE IS BETTER:

According to me, using Naive Bayes Classifier is better than Decision Tree Classifier
Reasons:
    1] Training Decision tree takes much more time than training Naive Bayes Classifier
    2] In decision tree, many words get pruned and hence accuracy for binary features get affected and reduces, while in Naive Bayes, we have the entire data for
    classification.
    3] Decision Tree takes more space as a matrix is formed for construction of tree. However, the Naive Bayes classifier only has a dicionary
    4] Naive Bayes classifier is more accurate than Decision tree in case of Spam Classification
"""

import os
import sys
import pickle
from collections import defaultdict
import math
import re
import heapq
class treeNode:
    def __init__(self, col = -1, val = None, res = None, l = None, r = None):
        self.col = col
        self.val = val
        self.res = res
        self.l = l
        self.r = r

#Checking for train techniques
def train(technique):
    # There are two techniques 1  Bayes and 2 Decision Tree
    techniques = {
        'bayes': train_bayes,
        'dt': train_dt
    }

    # Check if entered technique is valid
    if technique not in techniques:
        print "Invalid technique, exiting"
        exit()

    techniques[technique]()

#Checking for test techniques
def test(technique):
    # There are two techniques 1  Bayes and 2 Decision Tree
    techniques = {
        'bayes': test_bayes,
        'dt': test_dt
    }

    # Check if entered technique is valid
    if technique not in techniques:
        print "Invalid technique, exiting"
        exit()

    techniques[technique]()

#Training for Naive Bayes
def train_bayes():

    print "Training the model using Naive Bayes Classifier..."
    #Storing the directories
    spam_docs = os.listdir(spam_directory)
    notspam_docs = os.listdir(notspam_directory)

    #Taking counts of total documents
    spam_doc_count = len(spam_docs)
    notspam_doc_count = len(notspam_docs)

    #Counting continous frequency of words
    not_spam_dict = defaultdict(float)
    spam_dict = defaultdict(float)

    #Counting binary of words
    not_spam_dict_binary = defaultdict(float)
    spam_dict_binary = defaultdict(float)

    #Priors of spam and notspam
    prior_spam = float(spam_doc_count)/float(spam_doc_count + notspam_doc_count)
    prior_notspam = float(notspam_doc_count)/float(spam_doc_count + notspam_doc_count)

    #Creating the dictionaries for words in spam documents. Reading each spam file
    total_words_spam = 0
    total_words_spam_binary = 0
    for file_name in spam_docs:
        file = open(os.path.join(spam_directory, file_name))
        temp_list = re.split(" |\.",file.read().replace("\n","").strip())
        unique_spam = set(open(os.path.join(spam_directory, file_name)).read().split())

        for words in unique_spam:
            if (words.strip().lower() not in ignore_word_list):
                spam_dict_binary[words.strip().lower()] += 1
                total_words_spam_binary += 1

        for words in temp_list:
            if (words.strip().lower() not in ignore_word_list):
                spam_dict[words.strip().lower()] += 1
                total_words_spam += 1

    #Setting external values for total words, prior spam, spam document count
    spam_dict["TOTAL_WORDS"] = total_words_spam
    spam_dict["PRIOR_SPAM"] = prior_spam
    spam_dict_binary["TOTAL_WORDS"] = total_words_spam_binary
    spam_dict_binary["PRIOR_SPAM"] = prior_spam
    spam_dict_binary["SPAM_DOCS"] = spam_doc_count

    #Creating the dictionaries for words in nonspam documents. Reading each nonspam file
    total_words_notspam = 0
    total_words_notspam_binary = 0
    for file_name in notspam_docs:
        file = open(os.path.join(notspam_directory, file_name))
        temp_list = re.split(" |\.", file.read().replace("\n", "").strip())
        unique_notspam = set(open(os.path.join(notspam_directory, file_name)).read().split())

        for words in unique_notspam:
            if (words.strip().lower() not in ignore_word_list):
                not_spam_dict_binary[words.strip().lower()] += 1
                total_words_notspam_binary += 1

        for words in temp_list:
            if (words.strip().lower() not in ignore_word_list):
                not_spam_dict[words.strip().lower()] += 1
                total_words_notspam += 1

    # Setting external values for total words, prior nonspam, nonspam document count
    not_spam_dict["TOTAL_WORDS"] = total_words_notspam
    not_spam_dict["PRIOR_NOT_SPAM"] = prior_notspam
    not_spam_dict_binary["TOTAL_WORDS"] = total_words_notspam_binary
    not_spam_dict_binary["PRIOR_NOT_SPAM"] = prior_notspam
    not_spam_dict_binary["NOT_SPAM_DOCS"] = notspam_doc_count

    #Creating list for dumping in model file
    train_list = [spam_dict, not_spam_dict, spam_dict_binary, not_spam_dict_binary]
    pickle.dump(train_list, open(model_file, "wb"))

    #Removing external entries from list
    spam_dict.pop('TOTAL_WORDS')
    spam_dict.pop('PRIOR_SPAM')
    spam_dict_binary.pop('TOTAL_WORDS')
    spam_dict_binary.pop('PRIOR_SPAM')
    spam_dict_binary.pop('SPAM_DOCS')
    not_spam_dict.pop('TOTAL_WORDS')
    not_spam_dict.pop('PRIOR_NOT_SPAM')
    not_spam_dict_binary.pop('TOTAL_WORDS')
    not_spam_dict_binary.pop('PRIOR_NOT_SPAM')
    not_spam_dict_binary.pop('NOT_SPAM_DOCS')

    #Creating list of top 10 words that are MOST and LEAST associated with spam words for both binary and continous features
    top_10_spam = heapq.nlargest(10, spam_dict, spam_dict.get)
    top_10_notspam = heapq.nlargest(10, not_spam_dict, not_spam_dict.get)
    top_10_notspam_binary = heapq.nlargest(10, not_spam_dict_binary, not_spam_dict_binary.get)
    top_10_spam_binary = heapq.nlargest(10, spam_dict_binary, spam_dict_binary.get)

    #Printing the Output for trained Naive Bayes
    print "----"
    print "Top 10 words(Continous Features)"
    print "Top 10 words MOST associated with SPAM: ", top_10_spam
    print "Top 10 words LEAST associated with SPAM: ", top_10_notspam
    print "----"
    print "Top 10 words(Binary Features)"
    print "Top 10 words MOST associated with SPAM: ", top_10_notspam_binary
    print "Top 10 words LEAST associated with SPAM: ", top_10_spam_binary
    print "----"


def test_bayes():
    print "Testing Documents using Naive Bayes Classifier..."
    mylist = pickle.load(open(model_file, "rb"))
    spam_dict = mylist[0]
    not_spam_dict = mylist[1]
    spam_dict_binary = mylist[2]
    not_spam_dict_binary = mylist[3]

    spam_docs_test = os.listdir(spam_directory)
    notspam_docs_test = os.listdir(notspam_directory)

    #Initializing the counts to store the spam and nonspam classification
    notspam_spam_docs = 0.0
    notspam_notspam_docs = 0.0
    spam_spam_docs = 0.0
    spam_notspam_docs = 0.0

    notspam_spam_docs_binary = 0.0
    notspam_notspam_docs_binary = 0.0
    spam_spam_docs_binary = 0.0
    spam_notspam_docs_binary = 0.0

    print "Considering All NonSpam Documents**"
    print "----"

    for file_name in \
            notspam_docs_test:
        file = open(os.path.join(notspam_directory, file_name))
        temp_list = re.split(" |\.", file.read().replace("\n", "").strip())
        prob_spam = 0
        prob_notspam = 0
        prob_spam_binary = 0
        prob_notspam_binary = 0

        #Giving Prob = (1E-9) to words which are not in training set

        #Calculating for Continous Features
        for words in temp_list:
            if words.strip().lower() not in not_spam_dict:
                prob_notspam += math.log(1E-9)
            else:
                prob_notspam += math.log(float(not_spam_dict[words.strip().lower()])/float(not_spam_dict["TOTAL_WORDS"]))
        if prob_notspam != 0:
            prob_notspam += math.log(not_spam_dict["PRIOR_NOT_SPAM"])

        for words in temp_list:
            if words.strip().lower() not in spam_dict:
                prob_spam += math.log(1E-9)
            else:
                prob_spam += math.log(float(spam_dict[words.strip().lower()])/float(spam_dict["TOTAL_WORDS"]))
        if prob_spam != 0:
            prob_spam +=  math.log(spam_dict["PRIOR_SPAM"])

        if prob_spam > prob_notspam:
            notspam_spam_docs += 1
        else:
            notspam_notspam_docs += 1

        #Calculating for Binary Features
        for words in temp_list:
            if words.strip().lower() not in not_spam_dict_binary:
                prob_notspam_binary += math.log(1E-9)
            else:
                prob_notspam_binary += math.log(float(not_spam_dict_binary[words.strip().lower()])/float(not_spam_dict_binary["NOT_SPAM_DOCS"]))
        if prob_notspam_binary != 0:
            prob_notspam_binary += math.log(not_spam_dict_binary["PRIOR_NOT_SPAM"])

        for words in temp_list:
            if words.strip().lower() not in spam_dict_binary:
                prob_spam_binary += math.log(1E-9)
            else:
                prob_spam_binary += math.log(float(spam_dict_binary[words.strip().lower()])/float(spam_dict_binary["SPAM_DOCS"]))
        if prob_spam_binary != 0:
            prob_spam_binary +=  math.log(spam_dict_binary["PRIOR_SPAM"])

        if prob_spam_binary > prob_notspam_binary:
            notspam_spam_docs_binary += 1
        else:
            notspam_notspam_docs_binary += 1

    print "Continous Bag of Words"
    print "Notspam document considered as SPAM(FP): ", notspam_spam_docs
    print "Notspam document considered as NOTSPAM(TN): ", notspam_notspam_docs
    print "Accuracy", (notspam_notspam_docs) / float(((notspam_notspam_docs) + (notspam_spam_docs))) * 100
    print "----"

    print "Binary Bag of words"
    print "Notspam document considered as SPAM(FP): ",notspam_spam_docs_binary
    print "Notspam document considered as NOTSPAM(TN): ",notspam_notspam_docs_binary
    print "Accuracy Binary", (notspam_notspam_docs_binary)/float(((notspam_notspam_docs_binary)+(notspam_spam_docs_binary)))*100
    print "----"

    print "Considering All Spam Documents**"
    print "----"

    for file_name in spam_docs_test:
        file = open(os.path.join(spam_directory, file_name))
        temp_list = re.split(" |\.", file.read().replace("\n", "").strip())
        prob_spam = 0
        prob_notspam = 0

        #Calculating for Continous Features
        for words in temp_list:
            if words.strip().lower() not in not_spam_dict:
                prob_notspam += math.log(1E-9)
            else:
                prob_notspam += math.log(float(not_spam_dict[words.strip().lower()])/float(not_spam_dict["TOTAL_WORDS"]))
        if prob_notspam != 0:
            prob_notspam += math.log(not_spam_dict["PRIOR_NOT_SPAM"])

        for words in temp_list:
            if words.strip().lower() not in spam_dict:
                prob_spam += math.log(1E-9)
            else:
                prob_spam += math.log(float(spam_dict[words.strip().lower()])/float(spam_dict["TOTAL_WORDS"]))
        if prob_spam != 0:
            prob_spam += math.log(spam_dict["PRIOR_SPAM"])

        if prob_spam > prob_notspam:
            spam_spam_docs += 1
        else:
            spam_notspam_docs += 1

        # Calculating for Binary Features
        for words in temp_list:
            if words.strip().lower() not in not_spam_dict_binary:
                prob_notspam_binary += math.log(1E-9)
            else:
                prob_notspam_binary += math.log(float(not_spam_dict_binary[words.strip().lower()]) / float(
                    not_spam_dict_binary["NOT_SPAM_DOCS"]))
        if prob_notspam_binary != 0:
            prob_notspam_binary += math.log(not_spam_dict_binary["PRIOR_NOT_SPAM"])

        for words in temp_list:
            if words.strip().lower() not in spam_dict_binary:
                prob_spam_binary += math.log(1E-9)
            else:
                prob_spam_binary += math.log(
                    float(spam_dict_binary[words.strip().lower()]) / float(spam_dict_binary["SPAM_DOCS"]))
        if prob_spam_binary != 0:
            prob_spam_binary += math.log(spam_dict_binary["PRIOR_SPAM"])

        if prob_spam_binary > prob_notspam_binary:
            spam_spam_docs_binary += 1
        else:
            spam_notspam_docs_binary += 1

    print "Continuous Bag of Words"
    print "Spam document considered as SPAM(TP): ", spam_spam_docs
    print "Spam document considered as NOTSPAM(FN): ", spam_notspam_docs
    print "Accuracy", spam_spam_docs / float(((spam_notspam_docs) + (spam_spam_docs))) * 100
    print "----"

    print "Binary Bag of Words"
    print "Spam document considered as SPAM(TP): ", spam_spam_docs_binary
    print "Spam document considered as NOTSPAM(FN): ", spam_notspam_docs_binary
    print "Accuracy Binary", spam_spam_docs_binary/ float(((spam_notspam_docs_binary) + (spam_spam_docs_binary))) * 100
    print "----"

    #Generating Confusion Matrix for Continous Features
    print "Confusion Matrix for Continous Features**"
    matrix_continous = []
    col_value = [0] * 3
    col_value[0] = "  "
    col_value[1] = "SPAM"
    col_value[2] = "NOT SPAM"
    matrix_continous.append(col_value)
    row_value1 = [0] * 3
    row_value1[0] = "SPAM"
    row_value1[1] = spam_spam_docs
    row_value1[2] = spam_notspam_docs
    matrix_continous.append(row_value1)
    row_value2 = [0] * 3
    row_value2[0] = "NOT SPAM"
    row_value2[1] = notspam_spam_docs
    row_value2[2] = notspam_notspam_docs
    matrix_continous.append(row_value2)
    for values in matrix_continous:
        var1 = str(values[0])
        var2 = str(values[1])
        var3 = str(values[2])
        print var1.ljust(10, ' '), var2.ljust(10, ' '), var3.ljust(10, ' ')
    print "----"
    print "Total Accuracy for Classifier(Continous Features): ", (spam_spam_docs + notspam_notspam_docs) / float(
        spam_spam_docs + spam_notspam_docs + notspam_spam_docs + notspam_notspam_docs) * 100
    print "----"

    # Generating Confusion Matrix for Binary Features
    print "Confusion Matrix for Binary Features**"
    matrix_continous = []
    col_value = [0] * 3
    col_value[0] = "  "
    col_value[1] = "SPAM"
    col_value[2] = "NOT SPAM"
    matrix_continous.append(col_value)
    row_value1 = [0] * 3
    row_value1[0] = "SPAM"
    row_value1[1] = spam_spam_docs_binary
    row_value1[2] = spam_notspam_docs_binary
    matrix_continous.append(row_value1)
    row_value2 = [0] * 3
    row_value2[0] = "NOT SPAM"
    row_value2[1] = notspam_spam_docs_binary
    row_value2[2] = notspam_notspam_docs_binary
    matrix_continous.append(row_value2)
    for values in matrix_continous:
        var1 = str(values[0])
        var2 = str(values[1])
        var3 = str(values[2])
        print var1.ljust(10, ' '), var2.ljust(10, ' '), var3.ljust(10, ' ')
    print "----"
    print "Total Accuracy for Classifier(Binary Features): ", (spam_spam_docs_binary + notspam_notspam_docs_binary) / float(
        spam_spam_docs_binary + spam_notspam_docs_binary + notspam_spam_docs_binary + notspam_notspam_docs_binary) * 100
    print "----"


def train_dt():
    import time
    start = time.time()
    print "Training Decision Tree..."
    spam_docs = os.listdir(spam_directory)
    notspam_docs = os.listdir(notspam_directory)

    #Inititalizing the dictionary for count term frequency and list of unique words from spam and nonspam
    count_dict = defaultdict(int)
    unique_words = []

    # Reading Spam documents for unique words
    for file_name in spam_docs:
        file = open(os.path.join(spam_directory, file_name))
        temp_list = file.read().split()
        for word in temp_list:
            if word.isalpha() and word not in ignore_word_list and len(word) < 20:
                count_dict[word] += 1
            if word.isalpha() and word not in unique_words and word not in ignore_word_list and len(word) < 20:
                unique_words.append(word)

    # Reading NonSpam documents for unique words
    for file_name in notspam_docs:
        file = open(os.path.join(notspam_directory, file_name))
        temp_list = file.read().split()
        for word in temp_list:
            if word.isalpha() and word not in ignore_word_list and len(word) < 20:
                count_dict[word] += 1
            if word.isalpha() and word not in unique_words and word not in ignore_word_list and len(word) < 20:
                unique_words.append(word)

    unique_words = []

    for keys in count_dict:
        if count_dict[keys] > 10:
            unique_words.append(keys)
    print "Unique words list created"

    #Creating list of lists(Matrix) for generating tree
    my_data = []
    my_data_cont = []
    unique_words_len = len(unique_words)

    # Making matrix for Binary Features
    for file_name in spam_docs:
        document = [0] * (unique_words_len + 1)
        file = open(os.path.join(spam_directory, file_name))
        temp_list = file.read().split()
        for word in temp_list:
            if word in unique_words:
                index =unique_words.index(word)
                document[index] += 1
        document[unique_words_len] = "SPAM"
        my_data.append(document)

    for file_name in notspam_docs:
        document = [0] * (unique_words_len + 1)
        file = open(os.path.join(notspam_directory, file_name))
        temp_list = file.read().split()
        for word in temp_list:
            if word in unique_words:
                index =unique_words.index(word)
                document[index] += 1
        document[unique_words_len] = "NOTSPAM"
        my_data.append(document)

    print "List of list created(Binary)"

    #Making matrix for Continous Features
    for file_name in spam_docs:
        document = [0] * (unique_words_len + 1)
        file = open(os.path.join(spam_directory, file_name))
        temp_list = file.read().split()
        for word in temp_list:
            if word in unique_words:
                index = unique_words.index(word)
                document[index] += 1
        document[unique_words_len] = "SPAM"
        my_data_cont.append(document)

    for file_name in notspam_docs:
        document = [0] * (unique_words_len + 1)
        file = open(os.path.join(notspam_directory, file_name))
        temp_list = file.read().split()
        for word in temp_list:
            if word in unique_words:
                index = unique_words.index(word)
                document[index] += 1
        document[unique_words_len] = "NOTSPAM"
        my_data_cont.append(document)

    print "List of list created(Continous)"

    #Splitting the data into 2 different sets for calculating entropy
    def split(data, column, value):
        split_function = None
        split_function = lambda row: row[column] >= value

        #Creating left and right split of the data
        left_set = [row for row in data if split_function(row)]
        right_set = [row for row in data if not split_function(row)]

        return (left_set, right_set)

    def classlabels(data):
        #Counts the number of spam and non spam
        counts = {}
        for rows in data:
            row = rows[len(rows) - 1]
            if row not in counts: counts[row] = 0
            counts[row] += 1
        return counts

    def calc_entropy(data):
        from math import log
        log2 = lambda x: log(x) / log(2)
        results = classlabels(data)
        entropy = 0.0
        for key in results.keys():
            p = float(results[key]) / len(data)
            entropy = entropy - p * log2(p)
        return entropy

    def create_tree(data, entropy = calc_entropy):
        if len(data) == 0:
            return treeNode()

        criteria = None
        sets = None
        gain = 0.0

        current_score = entropy(data)

        column = len(data[0]) - 1
        for col in range(0, column):
            column_values = {}
            for row in data:
                column_values[row[col]] = 1
            for val in column_values.keys():
                (left, right) = split(data, col, val)
                # Calculating Information gain
                size = float(len(left)) / len(data)
                info_gain = current_score - size * entropy(left) - (1 - size) * entropy(right)
                if info_gain > gain and len(left) > 0 and len(right) > 0 :
                    gain = info_gain
                    criteria = (col, val)
                    sets = (left, right)

        # Creating inner branches of tree(expanding tree)
        if gain > 0:
            correct = create_tree(sets[0])
            incorrect = create_tree(sets[1])
            return treeNode(col = criteria[0], val = criteria[1], l = correct, r = incorrect)
        else:
            return treeNode(res = classlabels(data))

    def display_tree(tree, level, spacing = ' ', ):
        #Setting the level for displaying the tree(Our case 4)
        if level > 4:
            return
        if tree.res != None:
            print str(tree.res)
        else:
            #Here the function will print the correct and incorrect branches and split the feature
            print 'Node: ' + str(unique_words[tree.col])
            print spacing + 'Left ->',
            display_tree(tree.l, level + 1, spacing + '  ')
            print spacing + 'Right ->',
            display_tree(tree.r, level + 1, spacing + '  ')

    print "Building Tree"

    print "CREATING DECISION TREE(BINARY FEATURES)"
    tree = create_tree(my_data)

    print "CREATING DECISION TREE(CONTINOUS FEATURES)"
    tree_cont = create_tree(my_data_cont)

    #Printing and displaying the tree
    #1st Tree: For Binary Features
    #2nd Tree: For Continous Features

    print "Decision Tree for Binary Features"
    display_tree(tree, 1)
    print ' '
    print "Decision Tree for Continous Features"
    display_tree(tree_cont, 1)

    #Making the list for dumping into model file
    train_list = [unique_words, tree, tree_cont]
    pickle.dump(train_list, open(model_file, "wb"))

    print ' '
    print "Total time taken for training Decision Tree: ", time.time() - start, "seconds"

def test_dt():
    import time
    start = time.time()
    print "Testing Documents using Decision Tree"
    spam_docs = os.listdir(spam_directory)
    notspam_docs = os.listdir(notspam_directory)
    my_list = pickle.load(open(model_file, "rb"))
    unique_words = my_list[0]
    tree = my_list[1]
    tree_cont = my_list[2]

    def classification(document, tree):
        if tree.res != None:
            return tree.res
        else:
            value = document[tree.col]
            branch = None
            if value >= tree.val:
                branch = tree.l
            else:
                branch = tree.r
            return classification(document, branch)

    notspam_SPAM = 0
    notspam_NOTSPAM = 0
    spam_SPAM = 0
    spam_NOTSPAM = 0

    notspam_SPAM_cont = 0
    notspam_NOTSPAM_cont = 0
    spam_SPAM_cont = 0
    spam_NOTSPAM_cont = 0

    print "Considering Binary Features**"
    print "----"
    for file_name in notspam_docs:
        testdoc = [0] * len(unique_words)
        file = open(os.path.join(notspam_directory, file_name))
        temp = file.read().split()
        for word in range(0, len(unique_words)):
            if unique_words[word] in temp:
                testdoc[word] = 1

        if(classification(testdoc, tree).keys()[0] == "SPAM"):
            notspam_SPAM += 1
        else:
            notspam_NOTSPAM += 1

    print "Notspam document considered as SPAM(FP): ", notspam_SPAM
    print "Notspam document considered as NOTSPAM(TN): ", notspam_NOTSPAM
    print "Accuracy: ", (notspam_NOTSPAM/float(notspam_NOTSPAM + notspam_SPAM))*100
    print "----"


    for file_name in spam_docs:
        testdoc = [0] * len(unique_words)
        file = open(os.path.join(spam_directory, file_name))
        temp = file.read().split()
        for word in range(0, len(unique_words)):
            if unique_words[word] in temp:
                testdoc[word] = 1
        if (classification(testdoc, tree).keys()[0] == "SPAM"):
            spam_SPAM += 1
        else:
            spam_NOTSPAM += 1

    print "Spam document considered as SPAM(TP): ", spam_SPAM
    print "Spam document considered as NOTSPAM(FN): ", spam_NOTSPAM
    print "Accuracy: ", (spam_SPAM /float(spam_SPAM + spam_NOTSPAM)) * 100
    print "----"

    print "Confusion Matrix for Binary Features**"
    matrix_binary = []
    col_value = [0]*3
    col_value[0] = "  "
    col_value[1] = "SPAM"
    col_value[2] = "NOT SPAM"
    matrix_binary.append(col_value)
    row_value1 = [0]*3
    row_value1[0] = "SPAM"
    row_value1[1] = spam_SPAM
    row_value1[2] = spam_NOTSPAM
    matrix_binary.append(row_value1)
    row_value2 = [0] * 3
    row_value2[0] = "NOT SPAM"
    row_value2[1] = notspam_SPAM
    row_value2[2] = notspam_NOTSPAM
    matrix_binary.append(row_value2)
    for values in matrix_binary:
        var1 = str(values[0])
        var2 = str(values[1])
        var3 = str(values[2])
        print var1.ljust(10, ' '), var2.ljust(10, ' '), var3.ljust(10, ' ')
    print "----"
    print "Total Accuracy for Classifier(Binary Features): ", (spam_SPAM + notspam_NOTSPAM) /float(spam_SPAM + spam_NOTSPAM + notspam_SPAM + notspam_NOTSPAM) * 100
    print "----"

    print "Considering Continous Features**"
    print "----"
    for file_name in notspam_docs:
        testdoc = [0] * len(unique_words)
        file = open(os.path.join(notspam_directory, file_name))
        temp = file.read().split()
        for word in temp:
            if word in unique_words:
                index = unique_words.index(word)
                testdoc[index] += 1

        if(classification(testdoc, tree_cont).keys()[0] == "SPAM"):
            notspam_SPAM_cont += 1
        else:
            notspam_NOTSPAM_cont += 1

    print "Notspam document considered as SPAM(FP): ", notspam_SPAM_cont
    print "Notspam document considered as NOTSPAM(TN): ", notspam_NOTSPAM_cont
    print "Accuracy: ", (notspam_NOTSPAM_cont /float(notspam_SPAM_cont + notspam_NOTSPAM_cont)) * 100
    print "----"

    for file_name in spam_docs:
        testdoc = [0] * len(unique_words)
        file = open(os.path.join(spam_directory, file_name))
        temp = file.read().split()
        for word in temp:
            if word in unique_words:
                index = unique_words.index(word)
                testdoc[index] += 1

        if(classification(testdoc, tree_cont).keys()[0] == "SPAM"):
            spam_SPAM_cont += 1
        else:
            spam_NOTSPAM_cont += 1

    print "Spam document considered as SPAM(TP): ", spam_SPAM_cont
    print "Spam document considered as NOTSPAM(FN): ", spam_NOTSPAM_cont
    print "Accuracy: ", (spam_SPAM_cont /float(spam_SPAM_cont + spam_NOTSPAM_cont)) * 100
    print "----"

    print "Confusion Matrix for Continous Features**"
    matrix_continous = []
    col_value = [0] * 3
    col_value[0] = "  "
    col_value[1] = "SPAM"
    col_value[2] = "NOT SPAM"
    matrix_continous.append(col_value)
    row_value1 = [0] * 3
    row_value1[0] = "SPAM"
    row_value1[1] = spam_SPAM_cont
    row_value1[2] = spam_NOTSPAM_cont
    matrix_continous.append(row_value1)
    row_value2 = [0] * 3
    row_value2[0] = "NOT SPAM"
    row_value2[1] = notspam_SPAM_cont
    row_value2[2] = notspam_NOTSPAM_cont
    matrix_continous.append(row_value2)
    for values in matrix_continous:
        var1 = str(values[0])
        var2 = str(values[1])
        var3 = str(values[2])
        print var1.ljust(10, ' '), var2.ljust(10, ' '), var3.ljust(10, ' ')
    print "----"
    print "Total Accuracy for Classifier(Continous Features): ", (spam_SPAM_cont + notspam_NOTSPAM_cont) / float(spam_SPAM_cont + spam_NOTSPAM_cont + notspam_SPAM_cont + notspam_NOTSPAM_cont) * 100
    print "----"

    print "Total Time Taken: ", time.time() - start

if __name__ == "__main__":
    # validate arguments, exit if less than 4 arguments are passed
    if len(sys.argv) < 5:
        print "Please pass [mode] [technique] [dataset-directory] [model-file] as argument"
        print "Exiting now, please run the program again with correct arguments"
        exit()

    mode = sys.argv[1]
    technique = sys.argv[2]
    dataset_directory = sys.argv[3]
    model_file = sys.argv[4]

    # There are two modes, 1 Train and 2 Test
    modes = {
        'train': train,
        'test': test
    }

    if mode not in modes:
        print "Invalid mode, exiting"
        exit()

    # Check if train directory exists
    if not os.path.isdir(os.path.join(dataset_directory, "spam")):
        print "Invalid directory, exiting"
        exit()
    if not os.path.isdir(os.path.join(dataset_directory, "notspam")):
        print "Invalid directory, exiting"
        exit()

    spam_directory = os.path.join(dataset_directory, "spam")
    notspam_directory = os.path.join(dataset_directory, "notspam")

    # Storing words to be ignored in a dictionary
    ignore_word_list = {}
    for words in open("ignorewordlist.txt").readlines():
        ignore_word_list[words.strip()] = 1

    modes[mode](technique)