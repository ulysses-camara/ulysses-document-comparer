################
# BM25 e BM25L #
################

import numpy as np
import math
import copy

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess
from sklearn.metrics.pairwise import cosine_similarity

# Classes BM25 e BM25L
class BM25:
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size

        return nd

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_top_n_score(self, scores, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        top_n = np.argsort(scores)[::-1][:n]
        return [documents['name'][i] for i in top_n]
    
    def get_partial_score(self, query, documents):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        score = self.get_scores(query)

        return score

# BM25L
class BM25L(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus)

    #Calculo do IDF (Inverse Document Frequency)
    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps
    
    #Calculo do ctd
    def get_ctd(self, q_freq, b, doc_len, avg_len):
        ctd = q_freq/(1 - b + b*(doc_len)/(avg_len))
        return ctd

    #Avaliar a pontuacao de todos os documentos na base
    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)

        for q in query:
            
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = self.get_ctd(q_freq, self.b, doc_len, self.avgdl)
            score += (self.idf.get(q) or 0) * ( (ctd + 0.5) * (self.k1 + 1) /
                                               ( (ctd + 0.5) + self.k1 ))
        return score

    def lambda_update(self, scores, lambdas, data):
        new_scores = copy.deepcopy(scores)
        for i in range(len(data)):
            nome_doc = data["name"][i]#.strip()
            if nome_doc in lambdas.keys():
                new_scores[i] += lambdas[nome_doc]

        return new_scores
    
    def lambda_calc(self, queries_ds, query, cut, delta):
        all_queries = queries_ds["query"].tolist()
        all_queries.append(query)
        lista_doc = [eval(q) for q in queries_ds["user_feedback"]]
        vectorizer = TfidfVectorizer(tokenizer=preprocess)
        vectorizer.fit(all_queries)
        vsm_1 = vectorizer.transform([query])
        vsm_2 = vectorizer.transform(queries_ds["query"].tolist())
        similarities = cosine_similarity(vsm_1, vsm_2).tolist()[0]
        
        doc_sim = [(lista_doc[j], similarities[j]) for j in range(len(similarities)) if similarities[j] > cut]
        
        dic = {}
        for tup in doc_sim:
            for doc in tup[0]:
                if doc['class'] != 'i':
                    if doc['id'] in dic:
                        dic[doc['id']] += float(doc['score_normalized'])*float(tup[1])
                    else:
                        dic[doc['id']] = float(doc['score_normalized'])*float(tup[1])
                    
        for key in dic:
            dic[key] = np.log(dic[key] + 1) * delta
        
        return dic
