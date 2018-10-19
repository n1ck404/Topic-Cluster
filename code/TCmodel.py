#coding = utf-8
"""
# -*- coding: utf-8 -*-
# @Time    : 13/9/18 2:00 PM
# @Author  : Heng Guo
# @File    : TCmodel.py

"""

import os
import time
import sys
import collections
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import spacy
import heapq

import gensim
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases,Phraser
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models import TfidfModel
from gensim.models import HdpModel
from gensim.summarization.summarizer import summarize
from sklearn.decomposition import PCA

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import pyLDAvis
import pyLDAvis.gensim

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

"""

*　　　　　　　　┏┓　　　┏┓+ +
*　　　　　　　┏┛┻━━━┛┻┓ + +
*　　　　　　　┃　　　　　　　┃ 　
*　　　　　　　┃　　　━　　　┃ ++ + + +
*　　　　　　 ████━████ ┃+
*　　　　　　　┃　　　　　　　┃ +
*　　　　　　　┃　　　┻　　　┃
*　　　　　　　┃　　　　　　　┃ + +
*　　　　　　　┗━┓　　　┏━┛
*　　　　　　　　　┃　　　┃　　　　　　　　　　　
*　　　　　　　　　┃　　　┃ + + + +
*　　　　　　　　　┃　　　┃　　　　Code is far away from bug with the animal protecting　　　　　　　
*　　　　　　　　　┃　　　┃ + 　　　　神兽保佑,代码无bug　　
*　　　　　　　　　┃　　　┃
*　　　　　　　　　┃　　　┃　　+　　　　　　　　　
*　　　　　　　　　┃　 　　┗━━━┓ + +
*　　　　　　　　　┃ 　　　　　　　┣┓
*　　　　　　　　　┃ 　　　　　　　┏┛
*　　　　　　　　　┗┓┓┏━┳┓┏┛ + + + +
*　　　　　　　　　　┃┫┫　┃┫┫
*　　　　　　　　　　┗┻┛　┗┻┛+ + + +
"""

class TcModel:
    """
    Using gensim LDA model to implement the topic cluster
    """
    def __init__(self):
        self.original_data = []
        self.text = []
        self.token = []
        self.corpus = []
        self.id2word = []
        self.model_name = ''
        self.num_topics = 10
        self.iterations = 100
        self.model = None
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['be','say','-PRON-','ms','mr','year','cent','per','www','http','com'])



    def _tokenize_words(self,text):
        token = []
        total = len(text)
        for i in range(total):
            token.append(gensim.utils.simple_preprocess(text[i],deacc=True))
        return token


    def _phrase(self,token):
        bigram = Phrases(token,min_count=5,threshold=100)
        bigram_mod = Phraser(bigram)
        # trigram = Phrases(bigram_mod[token],min_count=5,threshold=100)
        # trigram_mod = Phraser(trigram)
        # return [trigram_mod[bigram_mod[doc]] for doc in token]
        return [bigram_mod[doc] for doc in token]


    def _lemmatization(self,token):
        nlp = spacy.load('en', disable=['parser', 'ner'],max_length=10000000)
        return_text = []
        #allow_postags = ['NOUN', 'ADJ', 'VERB', 'ADV','PROPN']
        allow_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        for i in token:
            sentence = nlp(" ".join(i))
            return_text.append([token.lemma_ for token in sentence if token.pos_ in allow_postags])
        return return_text

    def find_most_common(self,token,plot=False):
        word_list = []
        extra_stopwords = []
        for i in token:
            word_list.extend(i)
        word_dic = collections.Counter(word_list)
        #print(word_dic.most_common(100))
        tf = list(word_dic.values())
        tf.sort(reverse=True)
        if plot == True:
            print(tf[:100])
            plt.plot(range(500),tf[:500])
            plt.xlabel('word sequence')
            plt.ylabel('Term Frequency')
            plt.show()

        m_list = []
        for i in range(len(tf)-1):
            m_list.append(tf[i]-tf[i+1])
        k = tf[m_list.index(max(m_list))]
        print(k)
        k = 5000
        for i in word_dic:
            if word_dic[i] > k:
                extra_stopwords.append(i)

        print(extra_stopwords)
        return extra_stopwords

    def _remove_stopwords(self,token):
        return_text = []
        #self.stop_words.extend(self.find_most_common(token))
        for i in token:
            return_text.append([word for word in i if word not in self.stop_words])
        return return_text


    def _doc_topic(self):
        """
        Matrix = [doc_id,title,topic,probability,summary,content]
        """
        matrix = []
        for num in range(len(self.corpus)):
            row = self.model[self.corpus[num]]
            row = sorted(row,key=lambda x:x[1],reverse=True)
            for i,j in row:
                if float(j) < 0.05:
                    continue
                value = [self.original_data.ix[num]['id'],self.original_data.ix[num]['title'],i,j,self.original_data.ix[num]['summary'],self.original_data.ix[num]['content']]
                if value not in matrix:
                    matrix.append(value)

        matrix = pd.DataFrame(matrix,columns=['doc_id','title','topic','probability','summary','content'])
        self.doc_topic = matrix
        print(matrix)
        return matrix


    def _topic_doc(self):
        matrix = []
        for i in range(self.num_topics):
            doc_list = [i for i in list(self.doc_topic[self.doc_topic.topic == i].sort_values(by='probability',ascending=False)['doc_id'])]
            if doc_list == []:
                self.num_topics = i
                break
            output = ",".join([str(i) for i in doc_list])
            print('topic {}: {}'.format(i,output))
            matrix.append([i,output])
        return matrix


    def _readable_topic(self,sent_num = 3):
        output = []
        for i in range(self.num_topics):
            sent = ''
            content = []
            score_list = []
            topic_term = dict(self.model.show_topic(i,topn=10000))
            topic_list = self.doc_topic[self.doc_topic.topic == i]
            max_pro = heapq.nlargest(5,topic_list['probability'])
            for pro in max_pro:
                content.append(list(topic_list[topic_list.probability == pro]['content'])[0])
            content = ' '.join(content)
            
            content = [text for text in sent_tokenize(content)]
            for j in range(len(content)):
                words = gensim.utils.simple_preprocess(content[j],deacc=True)
                corpus = self.model.id2word.doc2bow(words)
                score = 0
                for word, num in corpus:
                    word = self.model.id2word.get(word)
                    if word in topic_term.keys():
                        score += num*topic_term[word]
                score_list.append(score)
            score_list = list(set(score_list))
            max_score = heapq.nlargest(sent_num,score_list)
            for j in range(len(max_score)):
                max_sent = score_list.index(max_score[j])
                print('topic {}: {}'.format(i,content[max_sent]))
                sent = sent + str('sentence {}: {}\n'.format(j+1,content[max_sent]))
            output.append([i,sent])
        return output

    def _topic_key(self):
        output = []
        for i in range(self.num_topics):
            output.append([i,','.join([item[0] for item in self.model.show_topic(i,topn=30)])])
        print(output)
        return output

    def train(self,path,num_topics=20,iterations=500,n_gram=True,lemmatization=True,stop_words=True, tfidf = True, model = 'lda'):
        """
        Trian the topic cluster model.
        Input value: data: pd.DataFrame format ['id','title','content','summary']
                     num_topics: (int) the number of topics
                     iterations: (int) total number of iteration times
        example:
        >>> lda = LDA_Model
        >>> lda.train(text)
        """
        data = load_data(str(path+'/output/data.csv'))
        self.original_data = data
        self.text = list(data['content'])
        self.num_topics = num_topics
        self.iterations = iterations
        self.model_name = model


        print('tokenizing...')
        self.token = self._tokenize_words(self.text)
        if stop_words == True:
            print('remove stop words...')
            self.token = self._remove_stopwords(self.token)

        if n_gram == True:
            print('phrasing...')
            self.token = self._phrase(self.token)

        if lemmatization == True:
            print('lemmatization...')
            self.token = self._lemmatization(self.token)

        self.id2word = Dictionary(self.token)
        self.corpus = [self.id2word.doc2bow(text) for text in self.token]
        if tfidf == True:
            print('calculate tfidf...')
            tfidf_model = TfidfModel(self.corpus)
            self.corpus = tfidf_model[self.corpus]

        if model == 'lda':
            self.model = LdaModel(corpus=self.corpus,
                                  id2word=self.id2word,
                                  num_topics=self.num_topics,
                                  iterations=self.iterations)
        if model == 'lsi':
            self.model = LsiModel(corpus=self.corpus, id2word=self.id2word, num_topics=self.num_topics)
        if model == 'hdp':
            self.model = HdpModel(corpus=self.corpus, id2word=self.id2word)
            self.num_topics = self.model.get_topics().shape[0]

        self.topic_key = pd.DataFrame(self._topic_key(), columns=['topic_id', 'key_words'])
        self.doc_topic = self._doc_topic()
        self.topic_doc = pd.DataFrame(self._topic_doc(), columns=['topic_id', 'document_id'])
        self.topic_sent = pd.DataFrame(self._readable_topic(), columns=['topic_id', 'most relative sentence'])

    def save(self, path = 'default'):
        #timestr = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        if path == 'default':
            path = 'model'
            try:
                os.mkdir(path)
            except:
                pass

        else:
            try:
                os.mkdir(path)
            except:
                pass

        if self.model_name == 'lda':
            self.model.save(str(path + '/lda.model'))
        if self.model_name == 'lsi':
            self.model.save(str(path + '/lsi.model'))
        if self.model_name == 'hdp':
            self.model.save(str(path + '/hdp.model'))

        f = open(str(path+'/original_data.pickle'),'wb')
        pickle.dump(self.original_data,f)
        f.close()
        f = open(str(path+'/text.pickle'),'wb')
        pickle.dump(self.text,f)
        f.close()
        f = open(str(path+'/token.pickle'),'wb')
        pickle.dump(self.token,f)
        f.close()
        f = open(str(path+'/corpus.pickle'),'wb')
        pickle.dump(self.corpus,f)
        f.close()
        path = path + '/result'
        self.save_result(path)

        avg,cosine_matrix = self.similarity()
        sns.set()
        label = []
        col = []
        for i in range(self.num_topics):
            cosine_matrix[i][i] = 0.5
            col.append('topic {}'.format(i))
        cosine_matrix = pd.DataFrame(cosine_matrix)
        cosine_matrix.columns = col
        cosine_matrix.index = col
        sns.heatmap(cosine_matrix,cmap='YlGnBu')
        plt.savefig(path+'/topic_similarity.jpg')
        cosine_matrix.to_csv(str(path+'/cosine_matrix.csv'))

    def save_result(self, path = 'default'):
        if path == 'default':
            path = 'model/result'
            try:
                os.mkdir(path)
            except:
                pass

        else:
            try:
                os.mkdir(path)
            except:
                pass

        # topic_key = pd.DataFrame(self.print_topics(num_topics=self.num_topics,num_words=10),columns=['topic id','key words'])
        # topic_key.to_csv(str(path+'/topic_key.csv'),index=False)
        # doc_topic = self._doc_topic()
        # doc_topic.to_csv(str(path+'/doc_topic.csv'))
        # topic_doc = pd.DataFrame(self._topic_doc(),columns=['topic id','document id'])
        # topic_doc.to_csv(str(path+'/topic_doc.csv'),index=False)
        # topic_sent = pd.DataFrame(self._readable_topic(),columns=['topic id','most relative sentence'])
        # topic_sent.to_csv(str(path+'/topic_sent.csv'),index=False)


        f = open(str(path+'/topic_key.pickle'),'wb')
        pickle.dump(self.topic_key,f)
        f.close()


        f = open(str(path+'/doc_topic.pickle'),'wb')
        pickle.dump(self.doc_topic,f)
        f.close()


        f = open(str(path+'/topic_doc.pickle'),'wb')
        pickle.dump(self.topic_doc,f)
        f.close()


        f = open(str(path+'/topic_sent.pickle'),'wb')
        pickle.dump(self.topic_sent,f)
        f.close()

    def load(self, path = 'default'):
        """
        :param path: the path of trained model.
        :return:
        """
        if path == 'default':
            path = 'model'
        file_list = os.listdir(path)
        for file in file_list:
            if file.endswith('.model'):
                self.model_name = file.split('.')[0]
        if self.model_name == 'lda':
            self.model = LdaModel.load(str(path+'/lda.model'))
        if self.model_name == 'lsi':
            self.model = LsiModel.load(str(path+'/lsi.model'))
        if self.model_name == 'hdp':
            self.model = HdpModel.load(str(path+'/hdp.model'))

        self.id2word = self.model.id2word
        if self.model_name == 'hdp':
            self.num_topics = self.model.get_topics().shape[0]
        else:
            self.num_topics = self.model.num_topics
        #self.iterations = self.model.iterations

        f = open(str(path+'/original_data.pickle'),'rb')
        self.original_data = pickle.load(f)
        f.close()
        f = open(str(path+'/text.pickle'),'rb')
        self.text = pickle.load(f)
        f.close()
        f = open(str(path+'/token.pickle'),'rb')
        self.token = pickle.load(f)
        f.close()
        f = open(str(path+'/corpus.pickle'),'rb')
        self.corpus = pickle.load(f)
        f.close()

        path = path+'/result'
        f = open(str(path + '/topic_key.pickle'), 'rb')
        self.topic_key = pickle.load(f)
        f.close()

        f = open(str(path + '/doc_topic.pickle'), 'rb')
        self.doc_topic = pickle.load(f)
        f.close()

        f = open(str(path + '/topic_doc.pickle'), 'rb')
        self.topic_doc = pickle.load(f)
        f.close()

        f = open(str(path + '/topic_sent.pickle'), 'rb')
        self.topic_sent = pickle.load(f)
        f.close()

        self.id2word = self.model.id2word
        if self.model_name == 'hdp':
            self.num_topics = self.topic_doc.shape[0]
        else:
            self.num_topics = self.model.num_topics

    def update(self,path,iterations=100,n_gram=True,lemmatization=True,stop_words=True,model = 'lda'):
        """
        :param path: The path of training file
        :param iterations: Only for lda model
        :param n_gram: choose if use n_gram feature, default is true
        :param lemmatization: choose if use lemmatization feature, default is true
        :param stop_words: choose if need to remove stop words, default is true
        :param model: choose what model to use, default is 'lda'
        :return:
        """
        data = load_data(path+'/output/data.csv')
        pd.concat([self.original_data,data],axis=0)
        text = list(data['content'])
        self.text.extend(text)

        print('tokenizing...')
        token = self._tokenize_words(text)
        self.token.extend(token)
        if n_gram == True:
            print('phrasing...')
            token = self._phrase(token)
            self.token.extend(token)
        if lemmatization == True:
            print('lemmatization...')
            token = self._lemmatization(token)
            self.token.extend(token)
        if stop_words == True:
            print('remove stop words...')
            token = self._remove_stopwords(token)
            self.token.extend(token)

        corpus = [self.id2word.doc2bow(text) for text in self.token]
        self.corpus.extend(corpus)
        self.model.update(corpus=corpus, iterations=iterations)

    def print_topics(self, num_topics=-1, num_words=10):
        """
        :param num_topics:(int, optional) – The number of topics to be selected
        :param num_words:(int, optional) – The number of words to be included per topics
        :return: list of (int, list of (str, float))
        """
        if num_topics == -1:
            num_topics = self.num_topics
        pprint.pprint(self.model.print_topics(num_topics=num_topics, num_words=num_words))
        return self.model.print_topics(num_topics=num_topics, num_words=num_words)

    def score(self):
        """
        Print the Coherence score of the model.

        """

        #print('\nPerplexity: ', self.model.log_perplexity(self.corpus))
        coherence_model_lda = CoherenceModel(model=self.model,
                                             texts=self.token,
                                             corpus=self.corpus,
                                             dictionary=self.id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

    def vis(self):
        """
        Visualization of the data through browser.
        """

        vis = pyLDAvis.gensim.prepare(self.model, self.corpus, self.id2word)
        pyLDAvis.show(vis)

    def consine(self,v1,v2):
        cosine = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        return cosine

    def similarity(self):
        topic_vector = self.model.get_topics()
        num_topics = topic_vector.shape[0]
        consine_matrix = np.diag(np.ones(num_topics))
        consine_list = []
        for i in range(num_topics-1):
            for j in range(i+1,num_topics):
                consine_matrix[i][j] = self.consine(topic_vector[i],topic_vector[j])
                consine_matrix[j][i] = consine_matrix[i][j]
                consine_list.append(consine_matrix[i][j])
        average = np.average(consine_list)
        return average,consine_matrix

    def to_gephi(self):
        _,cosine_matrix = self.similarity()
        edge = []
        for i in range(self.num_topics-1):
            for j in range(i+1,self.num_topics):
                edge.append(['topic {}'.format(i),'topic {}'.format(j),cosine_matrix[i][j]])

        for i in range(self.doc_topic.shape[0]):
            edge.append(['topic {}'.format(self.doc_topic.ix[i]['topic']),
                         self.doc_topic.ix[i]['doc_id'],
                         self.doc_topic.ix[i]['probability']])
        # edge = []
        # node = []
        # topic_vector = self.model.get_topics()

        #decomposition
    #     pca = PCA(n_components=1000)
    #     topic_vector = pca.fit_transform(topic_vector)
    #     print(len(topic_vector[0]))
    #     for i in range(len(topic_vector)):
    #         for j in range(len(topic_vector[i])):
    #             edge.append(['topic {}'.format(i),j,topic_vector[i][j]])
    #         node.append(['topic {}'.format(i),'topic {}'.format(i)])
    #
    #     return node,edge

    def to_neo4j(self,path):
        try:
            os.mkdir(path+'/database')
        except:
            pass

        path = path + '/database'
        self.original_data.to_csv(path+'/document.csv',index=False)
        topic = []
        relationship = []
        words = []
        for i in range(self.num_topics):
            topic.append(['topic {}'.format(i)])
            for word,pro in self.model.show_topic(i):
                words.append([word])
                relationship.append(['topic {}'.format(i),pro,word])

        topic = pd.DataFrame(topic)
        topic.columns = ['id']
        topic.to_csv(path+'/topic.csv',index=False)
        words = pd.DataFrame(words)
        words.columns = ['word']
        words.to_csv(path+'/words.csv',index = False)

        for i in range(len(self.doc_topic)):
            relationship.append(['topic {}'.format(self.doc_topic.ix[i]['topic']),self.doc_topic.ix[i]['probability'],self.doc_topic.ix[i]['doc_id']])

        _,consine_matrix = self.similarity()
        for i in range(self.num_topics-1):
            for j in range(i+1,self.num_topics):
                relationship.append(['topic %d' % i,consine_matrix[i][j],'topic %d' % j])

        relationship = pd.DataFrame(relationship)
        relationship.columns = ['source','probability','target']
        relationship.to_csv(path+'/relationship.csv',index=False)

        f = open(path+'/script.txt','w')
        f.write('load csv with headers from "file:///document.csv" as line \nmerge (d:Document{id:toInteger(line.id),title:line.title,summary:line.title,content:line.content})\n\n')
        f.write('load csv with headers from "file:///topic.csv" as line\nmerge (t:Topic{id:line.id})\n\n')
        f.write('load csv with headers from "file:///words.csv" as line\nmerge (w:Word{id:line.word})\n\n')
        f.write('load csv with headers from "file:///relationship.csv" as line\nmatch (from:Topic{id:line.source}),(to:Word{id:line.target})\nmerge (from)-[r:Key_word{probability:line.probability}]->(to)\n\n')
        f.write('load csv with headers from "file:///relationship.csv" as line\nmatch (from:Topic{id:line.source}),(to:Document{id:toInteger(line.target)})\nmerge (from)-[r:Include{probability:line.probability}]->(to)\n\n')
        f.write('load csv with headers from "file:///relationship.csv" as line\nmatch (from:Topic{id:line.source}),(to:Topic{id:line.target})\nmerge (from)<-[r:Similarity{probability:line.probability}]->(to)\n\n')
        f.close()

    def topic_content(self):
        """

        :return: list type, has K elements, K = the number of topics
        """
        content = []
        for i in range(self.num_topics):
            content.append(' '.join(list(self.doc_topic[self.doc_topic['topic'] == i].drop_duplicates('doc_id').sort_values('probability',ascending=False)['content'])))
        return content

    def key_sentence(self):
        content = self.topic_content()[:2]
        key_sent = []
        for i in content:
            key_sent.append(summarize(i,ratio=0.01))

        return key_sent

    def tfidf_keywords(self,topN = 10):
        output = []
        content = self.topic_content()
        token = self._tokenize_words(content)
        token = self._lemmatization(token)
        token = self._remove_stopwords(token)
        corpus = [self.id2word.doc2bow(text) for text in token]
        tfidf = TfidfModel(corpus)
        for i in range(self.num_topics):
            l = tfidf[corpus[i]]
            #l = corpus[i]
            l = sorted(l, key=lambda x: x[1], reverse=True)
            words = []
            for word,value in l:
                if len(words) == topN:
                    break
                else:
                    words.append(self.id2word.get(word))
            output.append([i,','.join(words)])
        return output

def load_data(path):
    print('Loding data...')
    data = pd.read_csv(path, encoding='utf-8')
    #docs = list(data['content'])
    return data


if __name__ == '__main__':
    st = time.time()

    #load original data
    path = '../data/alldata'

    model_name = 'lda'
    model = TcModel()
    model.train(path,num_topics=20,iterations=1000,model=model_name,n_gram=False)
    model.save(path=str('./'+ model_name))
    model.score()
    rt = time.time() - st
    print("total runing time = {}s".format(rt))
    #model.vis()
