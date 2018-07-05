from collections import defaultdict, Counter
from nltk.corpus import brown
import nltk
from sklearn.metrics import confusion_matrix
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle

# Keeps words and pos into a dictionary 
# where the key is a word and
# the value is a counter of POS and counts
word_tags = defaultdict(Counter)
pos_list = []
deno_tag_count = 0
word_list = []


list_of_files = os.listdir('brown/') # lists names of all files present in the brown/ folder.

temp_corpus = '' # Contains all lines from all files separated by \n

for file in list_of_files:  # There are around 500 files but I am just using 100 for the sake of making it small in scale
    with open('brown/' + file) as f:
        temp_corpus = temp_corpus + '\n' + f.read() # Here actual reading of content of file occured

corpus = temp_corpus.split('\n')

for line in corpus:
    if(len(line)>0):
        for word in line.split():
            try:            
                word, pos = word.split('/')
                
                word_tags[word][pos] +=1
                pos_list.append(pos)
                word_list.append(word)
                deno_tag_count += 1
                
            except: # If the w/tag form is not present by mistake
                break


'''
for word, pos in brown.tagged_words():
    word_tags[word][pos] +=1
    pos_list.append(pos)
    word_list.append(word)
    deno_tag_count += 1
'''
unique_words = set(word_list)
unique_tags = set(pos_list)

word2int = {}
int2word = {}

word2int = dict((w, i) for i, w in enumerate(unique_words))
int2word = dict((i, w) for i, w in enumerate(unique_words))

tag2int = {}
int2tag = {}

tag2int = dict((t, i) for i, t in enumerate(unique_tags))
int2tag = dict((i, t) for i, t in enumerate(unique_tags))


#print(len(unique_words))
#print(len(unique_tags))   
# Calculating p(w_i, t_i)
prob_word_tags = defaultdict(Counter)
print("Please wait for a minute ...")
#print("prob_word_tags starts here!")

for word in unique_words:
    total_count_temp = 0
    for tag in unique_tags:
        total_count_temp += word_tags[word][tag]
        
    for tag in unique_tags:
        prob_word_tags[word][tag] = word_tags[word][tag]/float(total_count_temp) 

#print("prob_word_tags ends here!")
#print(prob_word_tags)

# p(T_i) is being calculated!
prob_tags = {}
tags_count = Counter(pos_list)
for tag in tags_count:
    prob_tags[tag] = tags_count[tag]/float(deno_tag_count)

#prob_tags_sorted = sorted(prob_tags.items(), key=lambda pair: pair[1], reverse=True)

#print(prob_tags_sorted)
#print(tag_count)
#print(pos_list)
tag_tags = defaultdict(Counter)
for i in range(len(pos_list)-1):
    tag_tags[pos_list[i]][pos_list[i+1]] += 1
    tag_tags[pos_list[i+1]][pos_list[i]] += 1
    
    
# Computing p(T_i, T_i-1)
prob_tag_tags = defaultdict(Counter)
total_tag_Tags_count = 0
for tag in unique_tags:
    for tags in unique_tags:
        total_tag_Tags_count += tag_tags[tag][tags]
total_tag_Tags_count = total_tag_Tags_count/2
for tag in unique_tags:
    for tags in unique_tags:
        prob_tag_tags[tag][tags] = tag_tags[tag][tags]/ float(total_tag_Tags_count)
        
# HMM algorithm starts here ----------------------------------------------------

list_of_files = os.listdir('test/') # lists names of all files present in the brown/ folder.

temp_corpus = '' # Contains all lines from all files separated by \n

for file in list_of_files:  # There are around 500 files but I am just using 100 for the sake of making it small in scale
    with open('test/' + file) as f:
        temp_corpus = temp_corpus + '\n' + f.read() # Here actual reading of content of file occured

corpus = temp_corpus.split('\n')


sentences_words = [] # each element is a list of words present in a single sentence 
sentences_tags = [] # each element is a list of tags present in a single sentence

#We need words in each line separately as we are going to feed a sentence in the neural network not an individual word
for line in corpus:
    if(len(line)>0):
        words_in_a_line = [] #temporary storage for words in a line
        tags_in_a_line = [] #temporary storage for tags in a line
        for word in line.split():
            try:            
                w, tag = word.split('/')
            except: # If the w/tag form is not present by mistake
                break

            words_in_a_line.append(w.lower())
            tags_in_a_line.append(tag)
        
        sentences_words.append(words_in_a_line) 
        sentences_tags.append(tags_in_a_line)

# print(sentences_words[0])
# print(sentences_tags[0])


#vocab_words = set(sum(sentences_words, [])) #flattening of list is being done followed by finding unique words
#vocab_tags = set(sum(sentences_tags, [])) # This gives total number of tags 

# assert len(X_train) == len(Y_train)

#sentences_words_num = [[word2int[word] for word in sentence] for sentence in sentences_words]
#sentences_tags_num = [[tag2int[word] for word in sentence] for sentence in sentences_tags]

# print('sample X_train_numberised: ', sentences_words_num[0], '\n')
# print('sample Y_train_numberised: ', sentences_tags_num[0], '\n')

#X_train_numberised = np.asarray(sentences_words_num)
#Y_train_numberised = np.asarray(sentences_tags_num)



#print("Enter a sentence")
#print("Please avoid using any name in the sentnece!")
#s = input()

#sentence = nltk.word_tokenize(s)
'''
for word in sentence:
    for tag_i in prob_word_tags[word]:
        for tag_i_1 in prob_tag_tags[tag_i]:
            temp_p = (prob_word_tags[word][tag_i]*prob_tag_tags[tag_i][tag_i_1])/(prob_tags[tag_i]*prob_tags[tag_i_1])
'''




def  pos_tagger(sentence, idx, output, n):	
    # Base case, if whole sentence is traversed
    if (idx == n):
        #print(output)
        temp_p = 1.0
        for idx1, word in enumerate(sentence):
            if idx1==0:
                temp_p = temp_p*(output[idx1][1]*prob_tag_tags[output[idx1][0]]['.'])/(prob_tags[output[idx1][0]]*prob_tags['.'])    
            else:
                temp_p = temp_p*(output[idx1][1]*prob_tag_tags[output[idx1][0]][output[idx1-1][0]])/(prob_tags[output[idx1][0]]*prob_tags[output[idx1-1][0]]) 
        global maximum
        global result   
        if maximum < temp_p:
            maximum = temp_p
            result = output
           
        
        return result
 
    # Try all possible tags for current word in sentence and recur for remaining words
    for tag, prob in dict_of_word_tags_prob[sentence[idx]].items():
        output[idx] = (tag, prob)
        return pos_tagger(sentence, idx+1, output, n)






y_pred_tag = []
y_actual_tag = []
line_no = 1

test_s = []

for idx1, sentence in enumerate(sentences_words):

    print(line_no)
    line_no += 1
  
            
    prob_word_tags_sentence = defaultdict(Counter)
    for word in sentence:
        for tag in unique_tags:
            prob_word_tags_sentence[word][tag] = prob_word_tags[word][tag]    
    
    
    #def pos_tagger(sentence, idx, n):

    dict_of_word_tags_prob = {}
    for word in sentence:
	    dict_of_word_tags_prob[word] = ( dict((k, v) for k, v in prob_word_tags_sentence[word].items() if v > 0.0))

    #for word in sentence:
    #    print(dict_of_word_tags_prob[word])

    n = len(sentence)

    result = []
    maximum = -1
   
    ans = pos_tagger(sentence, 0, [("",0.0)]*n, n)
    out = []
    #print(ans)
    if ans is None:
        continue
        
    test_s.append(sentence)
    
    for idx, ele in enumerate(ans):
        #out.append((sentence[idx], ele[0]))
        out.append(ele[0])
    y_pred_tag.append(out)
    y_actual_tag.append(sentences_tags[idx1])
    #print(out)
   
y_pred_tag = sum(y_pred_tag,[]) 
#y_pred_tag_num = [tag2int[tag] for tag in y_pred_tag]
y_actual_tag = sum(y_actual_tag,[]) 


#y_actual_tag_num = [tag2int[tag] for tag in y_actual_tag]
#print(y_pred_tag)

frequent_pos = []
for pos, count in Counter(y_actual_tag).most_common(12):
    frequent_pos.append(pos)

frequent_pos.remove('.')
frequent_pos.remove(',')

print("10 Most frequent actual tags!")
print(frequent_pos)

actual = []
predicted = []

for idx2, pos in enumerate(y_actual_tag):
    if pos in frequent_pos:
        actual.append(pos)
        predicted.append(y_pred_tag[idx2])

temp_labels = sorted(list(set(actual+predicted)))

#print(temp_labels)
cm = confusion_matrix(actual, predicted, temp_labels)
cm = np.ndarray.tolist(cm)

#for idx5, row in enumerate(cm):
#    for idx6, ele in enumerate(row):
#        cm[idx5][idx6] = "    "+str(ele)

cm.insert(0, temp_labels)

for idx3, row in enumerate(cm):
    if idx3==0:
        row.insert(0,'*')
    else:
        row.insert(0,temp_labels[idx3])
    

print(cm)


pickle_files = [test_s, y_actual_tag, frequent_pos]
if not os.path.exists('test_pickle_file/'):
    print('test_pickle_file/ is created to save pickled actual labels and test sentences!')
    os.makedirs('test_pickle_file/')

with open('test_pickle_file/test_data.pkl', 'wb') as f:
    pickle.dump(pickle_files, f)

print('Saved as pickle file')



#print(tag_tags["JJ"])
# To access the POS counter.    
#print (word_tags['Red'])
#print (word_tags['Marlowe'])

#Greatest number of distinct tag.
#word_with_most_distinct_pos = sorted(word_tags, key=lambda x: len(word_tags[x]), reverse=True)[0]

#print (word_with_most_distinct_pos)
#print (word_tags[word_with_most_distinct_pos])
#print (len(word_tags[word_with_most_distinct_pos]))
