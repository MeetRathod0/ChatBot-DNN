import numpy as np
import pickle
import json
import random
import tflearn
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Chatbot modal trainer, call train method from class Trainer
class Trainer:
    def train(self):
        with open('intents.json') as file:
            data = json.load(file)
        #print(data)

        # Some cleaning of data in intents.json
        stemmed_words = []
        tags = []
        ignore_words = ['!', '?', '.']
        corpus = []

        for intent in data['intents']:
            for pattern in intent['patterns']:
                stemmed_pattern = Utils().clean_pattern(pattern, ignore_words)
                stemmed_words.extend(stemmed_pattern)
                corpus.append((stemmed_pattern, intent['tag']))
            if intent['tag'] not in tags:
                tags.append(intent['tag'])

        # remove duplicates and sort
        stemmed_words = sorted(list(set(stemmed_words)))
        tags = sorted(list(set(tags)))

        #print(stemmed_words)
        #print(tags)
        #print(corpus)

        # Creating numeric features and labels out of cleaned data
        X = []
        y = []
        for item in corpus:
            bag = [] #array of 1 and 0. 1 if stemmed word is present in stemmed pattern
            stemmed_pattern = item[0]
            for w in stemmed_words:
                if w in stemmed_pattern:
                    bag.append(1)
                else:
                    bag.append(0)

            tags_row = [] #array of 1 and 0. 1 for current tag and for everything else 0.
            current_tag = item[1]
            for tag in tags:
                if tag == current_tag:
                    tags_row.append(1)
                else:
                    tags_row.append(0)

            #for each item in corpus, X will be array indicating stemmed words and y array indicating tags
            X.append(bag)
            y.append(tags_row) 

        X = np.array(X)
        y = np.array(y)
        #print(X)
        #print(y)

        # saving variables in pickle to be used by main.py
        with open('saved_variables.pickle', 'wb') as file:
            pickle.dump((stemmed_words, tags, ignore_words, X, y), file)

        model = Utils().define_network(X, y)
        model.fit(X, y, n_epoch=1120, batch_size=8, show_metric=True) 
        model.save("chatbot_model.tflearn")

# data cleaning & DNN
class Utils:
    def __init__(self):
        self.stemmer = LancasterStemmer() #stemmer to get stem of a word. ex. 'say' would be stem word of 'saying'.

    def define_network(self,X, y):
        tf.compat.v1.reset_default_graph() #Clears the default graph stack and resets the global default graph
        # neural network's layers
        network = tflearn.input_data(shape= [None, len(X[0])]) #input layer
        network = tflearn.fully_connected(network, 8) #1st hidden layer
        network = tflearn.fully_connected(network, 8) #2nd hidden layer
        network = tflearn.fully_connected(network, len(y[0]), activation= 'softmax') #output layer
        network = tflearn.regression(network)
        model = tflearn.DNN(network, tensorboard_dir='tflearn_logs') #tensorboard_dir is path to store logs
        return model

    # gives stemmed, tokenized words list from sentence pattern without words in ignore_words list
    def clean_pattern(self,pattern, ignore_words):
        stemmed_pattern = []
        wrds = nltk.word_tokenize(pattern)
        for w in wrds:
            if w not in ignore_words:
                stemmed_pattern.append(self.stemmer.stem(w.lower()))
        return stemmed_pattern

    # generates a numpy array of 0 & 1 from string sentence of user to fed to model
    def bag_of_words(self,sentence, stemmed_words, ignore_words):
        bag = []
        stemmed_pattern = self.clean_pattern(sentence, ignore_words)
        for w in stemmed_words:
            if w in stemmed_pattern:
                bag.append(1)
            else:
                bag.append(0)
        return np.array(bag)

# Edith class:
class EChatBot:
    def __init__(self):
        with open('saved_variables.pickle', 'rb') as file:
            self.stemmed_words, self.tags, self.ignore_words, self.X, self.y = pickle.load(file) 

        with open('intents.json') as file:
            self.data = json.load(file)


        self.model = Utils().define_network(self.X, self.y)
        self.model.load("chatbot_model.tflearn")

    # to handle previous context and give advantage to results of that context
    def context_func(self,context, user_input):
        model_input = [Utils().bag_of_words(user_input, self.stemmed_words, self.ignore_words)]
        results = self.model.predict(model_input)[0]
        for intent in self.data['intents']:
            if 'context_filter' in intent:
                if intent['context_filter'] == context:
                    # looping through tags and their indices
                    for tg_index, tg in enumerate(self.tags):
                        if tg == intent['tag']:
                            results[tg_index] += 0.5 
        return results

    def chat(self,user_input,show_tags_probability= False):
        probability_threshold = 0.4 
        context = ""
        user_input = user_input.lower()
        # if context from previous response is there, results of that context gets advantage
        if context:
            results = self.context_func(context, user_input)
        else:
            model_input = [Utils().bag_of_words(user_input, self.stemmed_words, self.ignore_words)] #as model is trained on 2d array
            results = self.model.predict(model_input)[0] #gives array of probabilities for all tags
        # to show probabilities given by model for each tags
        if show_tags_probability:
            probability_dict = {}
            for i, j in zip(self.tags, results):
                probability_dict[i] = j
            print('tags_probabilities: {}'.format(probability_dict))
            probability_dict.clear()

        context = "" #reset the context
        result_index = np.argmax(results) #to get index of max probability
        
        #to filter out predictions below threshold
        if results[result_index] > probability_threshold: 
            tag = self.tags[result_index] #tag associated with user_input according to model predicition
            for intent in self.data['intents']:
                if intent['tag'] == tag:
                    response = random.choice(intent['responses'])
                    # check if context is set for current intent
                    if 'context_set' in intent:
                        context = intent['context_set']
                    break
        else:
            response = "I didn't understand!..."
        
        return {"Response":response}

print(EChatBot().chat("Hi"))
# for train chatbot:
# write intents in intents.json file & call Trainer().train() method to train chatbot.
# call chatbot use EChatBot().chat(<msg>) # class EChatBot, method chat