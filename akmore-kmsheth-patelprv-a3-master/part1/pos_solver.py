###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)

"""Report
1. Estimating P(S1), P(Si+1|Si) and P(wi|si)

p(S1)= count(Total number of times Si appeared in the first porsition)/count(total number of sentences)

Transition Probility:
For transition probability, we have a dictionary of dictionary.
P(Si+1|Si)= P(Si,Si+1)/P(si)

Emission probability:
P(Wi|Si) = count(Wi, Si) / count(Si)

Also calculated P(part of speech)= count(Si)/count(Total Part of speech count)

We have used dictionary for storing for the following probabilites:
P(Si+1|Si) - dict_transition_prob[si1][si]
P(S1) - dict_initial_prob[i]
P(Wi|Si) - dict_emission_prob[wi][si]

Navie Bayes:

Here we are selecting a POS by finding most probable tag for each word.

HMM:

We first calculated the intial probability of a particular part of speech starting the sentence
We are using the dynamic programming approach to calculate the probabilities at each state. We calculate the probabilities of previous state and store it into a dictionary.
We then estimate the next value by multiplying the emission value of the current word with the maximum product of value of last state with transition probability of current state with each pos
At each state the part of speech with maximum probability is stored.
Thus to get the most probable sequence we are selecting values of Part of Speech based on Part of speech tagging based on the Part of speech tagged at current state using its stored value


Variable Elimination:

Variable elimination becomes a bit similar to Viterbi calculation. In this Complex algorithm, we are using dynamic approach to store the probabilities of previous states from left.
At each point we are evaluating the previous states and finding the maximum transition probability from all previous states
We return the list of most probable state sequence


2. Results
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
     1. Simplified:       93.92%               47.45%
            2. HMM:       87.75%               35.80%
        3. Complex:       89.03%               37.30%

3. Challenges and Assumptions
Assumptions:
1. Setting probabilities of unknown words to a very small value:
Here we were facing challenges when any new word comes which is not present in the training corpus. For such words we have set their probability to a very small value(1e-10).

2. Assigning Transition values for unknown transitions to a very small probability:
While learning the corpus, if there does not exists any transition from some Part Of Speech to another then we are assigning
its value as 1e-10 so that keeping a small probability of its happening.

3. If the word is not present in the training data then we are assigning the most probable part of speech from the data set to such data.

4. Each sentence takes 2 seconds to run"""

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:
    dict_emission_prob = {}
    dict_pos_prob = {}
    dict_initial_prob = {}
    pos_tags = []
    dict_transition_prob = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        global dict_emission_prob
        global dict_pos_prob
        sum = 0.0
        mult = math.log(1)
        for i in xrange(0, len(sentence)):
            if sentence[i] in dict_emission_prob:
                emission = dict_emission_prob[sentence[i]][label[i]]
                if emission == 0:
                    emission = 1e-50
                sum = sum + (emission * dict_pos_prob[label[i]])
                mult = mult + math.log(emission * dict_pos_prob[label[i]])
                result = mult - math.log(sum)
            else:
                result = 1
        return result

    # Do the training!
    def train(self, data):
        global dict_emission_prob
        global dict_pos_prob
        global pos_tags
        global dict_pos_prob
        global dict_initial_prob
        global dict_transition_prob
        pos = []  # list to store the parts of speech
        words = []  # list to store the words
        value_pos = []
        value_word = []
        dict_pos_count = {}
        pos_tags = ['noun', 'det', 'adj', 'verb', 'adp', '.', 'conj', 'adv', 'prt', 'num', 'pron',
                    'x']  # list storing all the parts of speech
        pos_prob = []  # List storing all the probabilities of the part of speech
        dict_pos_prob = {}  # Dictionary to store the part of speech tag and the individual part of speech probability
        total_word_count = 0  # total number of words in the training data
        det_count, noun_count, adj_count, verb_count, adp_count, pun_count, conj_count, adv_count, prt_count, num_count, pron_count, foreign_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        prob_det, prob_noun, prob_adj, prob_verb, prob_adp, prob_pun, prob_conj, prob_adv, prob_prt, prob_num, prob_pron, prob_foreign = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for line in data:
            words += line[0::2]
            pos += line[1::2]

        for element in pos:
            value_pos += element

        for element in words:
            value_word += element

        for i in pos_tags:
            dict_pos_count[i] = 0

        for element in value_pos:
            dict_pos_count[element] += 1

        total_word_count = float(len(value_pos))
        for posTag in pos_tags:
            total_prob_tag = dict_pos_count[posTag] / total_word_count
            pos_prob.append(total_prob_tag)

        dict_pos_prob = dict(zip(pos_tags, pos_prob))  # Stores the pos tag and the individual pos probability in a dict
        dict_transition_prob = {}  # To store the transition prob eg {'noun':{'noun':0.375,'verb':'0.99',....},'verb':{'noun':0.77}}

        for si in pos_tags:
            dict_transition_prob[si] = {}
            for si1 in pos_tags:
                dict_transition_prob[si][si1] = 0  # Assigns a value of 0 to the transition prob nested dict
        new_dict_trans={}
        for si in pos_tags:
            new_dict_trans[si] = {}
            for si1 in pos_tags:
                new_dict_trans[si][si1] = 0  # Assigns a value of 0 to the transition prob nested dict

        dict_trans_count={}
        dict_trans_count = dict(zip(pos_tags, [0] * len(pos_tags)))

        for i in pos:
            for j in range(0, len(i) - 1):
                dict_transition_prob[i[j]][i[j + 1]] += 1
                if i[j]=='noun':
                    dict_trans_count[i[j]]+=1
                elif i[j] == 'det':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'adj':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'verb':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'adp':
                    dict_trans_count[i[j]] += 1
                elif i[j] == '.':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'conj':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'adv':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'prt':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'num':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'pron':
                    dict_trans_count[i[j]] += 1
                elif i[j] == 'x':
                    dict_trans_count[i[j]] += 1
                else:
                    continue

        for si in pos_tags:
            for si1 in pos_tags:
                dict_transition_prob[si1][si] /= float(dict_trans_count[si1])

        for si in pos_tags:
            for si1 in pos_tags:
                if dict_transition_prob[si1][si] == 0:
                    dict_transition_prob[si1][si] = 1e-10

        dict_initial_prob = dict(zip(pos_tags, [0] * len(pos_tags)))
        for i in pos:
            for j in range(0, len(i)):
                dict_initial_prob[i[j]] += 1
                break

        for i in dict_initial_prob.keys():
            dict_initial_prob[i] /= float(len(pos))

        dict_unique_words = list(set(value_word))
        dict_emission_prob = dict(zip(pos_tags, [0] * len(pos_tags)))

        for si in dict_unique_words:
            dict_emission_prob[si] = {}
            for si1 in pos_tags:
                dict_emission_prob[si][si1] = 0  # Assigns a value of 0 to the transition prob nested dict
        temp_list = []

        for i, j in zip(words, pos):
            for value_word, value_pos in zip(i, j):
                variable = (value_word, value_pos)
                temp_list.append(variable)

        for i in temp_list:
            for j in range(0, len(i) - 1):
                dict_emission_prob[i[j]][i[j + 1]] += 1

        for wi in dict_unique_words:
            for si in pos_tags:
                dict_emission_prob[wi][si] /= float(
                    dict_pos_count[si])  # Calculates the transition prob and stores it in the dictionary

    # Functions for each algorithm.
    def simplified(self, sentence):
        global dict_emission_prob  # to store the emission prob
        global dict_pos_prob
        global pos_tags
        global max_pos_prob
        # new_dict_emission_prob = {}
        from copy import deepcopy
        new_dict_emission_prob = deepcopy(dict_emission_prob)
        pos_list = []  # to store part of speech of probable sequence
        pos_prob = []  # to store the probabilities of the probable sequence

        # Calculating prob of P(w|s) * P(s)
        for word in sentence:
            if word in new_dict_emission_prob:
                for tag in pos_tags:
                    new_dict_emission_prob[word][tag] = dict_emission_prob[word][tag] * dict_pos_prob.get(tag)
            else:
                continue

        for each_pos in dict_pos_prob:
            max_pos_prob = max(dict_pos_prob, key=dict_pos_prob.get)

        for word in sentence:
            if word in new_dict_emission_prob:
                max_pos = max(new_dict_emission_prob[word], key=new_dict_emission_prob[
                    word].get)  # assigning the max probable pos to max_pos variable
                prob_value = round(new_dict_emission_prob[word][max_pos],3)  # assigning the max probable probability to prob_value variable
                pos_list.append(max_pos)
                pos_prob.append(prob_value)
            else:
                pos_list.append(
                    max_pos_prob)  # assigns the most probable pos to the word which is not present in the test data
                pos_prob.append(1e-10)  # assigns a random probability to the word which is not present in the test data
        # return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]
        return [pos_list], [pos_prob]

    def hmm(self, sentence):
        global dict_emission_prob
        global dict_initial_prob
        global pos_tags
        global dict_transition_prob
        global max_pos_prob
        list = []
        value_list = []
        dict_viterbi = {}

        for word in sentence:
            dict_viterbi[word] = {}
            for tag in pos_tags:
                dict_viterbi[word][tag] = 0
        word = sentence[0]

        if word in dict_emission_prob:
            for tag in pos_tags:
                if (dict_emission_prob[word][tag] * dict_initial_prob[tag]) == 0:
                    dict_viterbi[word][tag] = 1e-10
                else:
                    dict_viterbi[word][tag] = dict_emission_prob[word][tag] * dict_initial_prob[tag]

        for i in range(1, len(sentence)):
            current_word = sentence[i]
            previous_word = sentence[i - 1]
            max_prob = 0
            if current_word in dict_emission_prob:
                for current_tag in pos_tags:
                    for previous_tag in pos_tags:
                        if max_prob < dict_viterbi[previous_word][previous_tag] * dict_transition_prob[previous_tag][
                            current_tag]:
                            max_prob = dict_viterbi[previous_word][previous_tag] * dict_transition_prob[current_tag][
                                previous_tag]
                    if (dict_emission_prob[current_word][current_tag] * max_prob) == 0:
                        dict_viterbi[current_word][current_tag] = 1e-100
                    else:
                        dict_viterbi[current_word][current_tag] = (
                            dict_emission_prob[current_word][current_tag] * max_prob)
            else:
                dict_viterbi[current_word][max_pos_prob] = 1e-10

        for word in sentence:
            max_pos_prob_viterbi = max(dict_viterbi[word], key=dict_viterbi[word].get)
            list.append(max_pos_prob_viterbi)
            # value_list.append(dict_viterbi[word][max_pos_prob_viterbi])
        # return [ [ [ "noun" ] * len(sentence)], [] ]
        return [list], []

    def complex(self, sentence):
        global dict_emission_prob
        global dict_initial_prob
        global pos_tags
        global dict_transition_prob
        global max_pos_prob
        list = []
        value_list = []
        dict_var_elimination = {}

        for word in sentence:
            dict_var_elimination[word] = {}
            for tag in pos_tags:
                dict_var_elimination[word][tag] = 0

        if sentence[0] in dict_emission_prob:
            for tag in pos_tags:
                if (dict_emission_prob[sentence[0]][tag] * dict_initial_prob[tag]) == 0:
                    dict_var_elimination[sentence[0]][tag] = 1e-10
                else:
                    dict_var_elimination[sentence[0]][tag] = (
                        dict_emission_prob[sentence[0]][tag] * dict_initial_prob[tag])

        if len(sentence) > 1:
            if sentence[1] in dict_emission_prob:
                current = sentence[1]
                previous = sentence[0]
                max_prob = 0
                for current_tag in pos_tags:
                    for previous_tag in pos_tags:
                        if max_prob < (
                                    dict_var_elimination[previous][previous_tag] * dict_transition_prob[previous_tag][
                                    current_tag]):
                            max_prob = dict_var_elimination[previous][previous_tag] * \
                                       dict_transition_prob[previous_tag][
                                           current_tag]
                        if (dict_emission_prob[current][current_tag] * max_prob) == 0:
                            dict_var_elimination[current][current_tag] = 1e-100
                        else:
                            dict_var_elimination[current][current_tag] = (
                                dict_emission_prob[current][current_tag] * max_prob)
            else:
                dict_var_elimination[sentence[1]][max_pos_prob] = 1e-10

        for i in range(2, len(sentence)):
            current1 = sentence[i]
            previous1 = sentence[i - 1]
            previous2 = sentence[i - 2]
            max1, max2, maxprob = 0, 0, 0
            if sentence[i] in dict_emission_prob:
                for currentTag1 in pos_tags:
                    for previousTag1 in pos_tags:
                        if max1 < (dict_var_elimination[previous1][previousTag1] * dict_transition_prob[previousTag1][
                            currentTag1]):
                            max1 = (dict_var_elimination[previous1][previousTag1] * dict_transition_prob[previousTag1][
                                currentTag1])

                    for previousTag2 in pos_tags:
                        if max2 < (dict_var_elimination[previous2][previousTag2] * dict_transition_prob[previousTag2][
                            currentTag1]):
                            max2 = (dict_var_elimination[previous2][previousTag2] * dict_transition_prob[previousTag2][
                                currentTag1])

                    maxprob = max1 if max1 > max2 else max2

                    if (dict_emission_prob[current1][currentTag1] * maxprob) == 0:
                        dict_var_elimination[current1][currentTag1] = 1e-100
                    else:
                        dict_var_elimination[current1][currentTag1] = (
                            dict_emission_prob[current1][currentTag1] * maxprob)
            else:
                dict_var_elimination[current1][max_pos_prob] = 1e-10

        for word in sentence:
            max_pos_prob_var = max(dict_var_elimination[word], key=dict_var_elimination[word].get)
            list.append(max_pos_prob_var)
            max_prob_value=round((dict_var_elimination[word][max_pos_prob_var]),5)
            value_list.append(max_prob_value)
        # return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

        return [list], [value_list]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        print "Inside the solver class"
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"
