import math
import numpy as np
import sys
from multiprocessing import Pool, cpu_count
from improve_similarity import lambda_update, lambda_calc

DEFAULT_CUT = 0.4
DEFAULT_DELTA = 0.2
DEFAULT_PESO_POUCO_RELEVANTES = 0.2

class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self.term_freqs = {} # dicionário para guardar os "term frequencies" (TF), da forma {'termo': [lista de frequências onde freq > 0], ...}
        self.term_docs = {} # dicionário para guardar os documentos onde ocorre cada termo {'termo': [lista de docs onde ocorre 'termo'], ...}

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        num_doc = 0
        doc_n = 0  # nº do documento
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            # dicionário para controlar quebra de documento para cada n-grama
            doc_changed = {}
            for word in document:
                doc_changed[sys.intern(word)] = True # significa que mudou o documento; indica para todas as palavras do documento

            for word in document:
                # Incrementa o dicionário de TF
                if word not in self.term_freqs:
                    self.term_freqs[sys.intern(word)] = [1]
                    self.term_docs[sys.intern(word)] = [doc_n]
                    doc_changed[sys.intern(word)] = False
                else:
                    if not doc_changed[sys.intern(word)]:
                        self.term_freqs[sys.intern(word)][-1] += 1  # incrementa frequência do termo no documento = TF(t, d)
                    else:
                        self.term_freqs[sys.intern(word)].append(1)
                        self.term_docs[sys.intern(word)].append(doc_n)
                        doc_changed[sys.intern(word)] = False
            doc_n += 1

        # monta o dicionário DF (Document Frequency)
        df = {}
        for term_freq, lst_freqs in zip(self.term_freqs.keys(), self.term_freqs.values()):
            df[sys.intern(term_freq)] = len(lst_freqs)

        self.avgdl = num_doc / self.corpus_size
        return df

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()


#Implementacao do BM25L - adapta parametros para corrigir a preferencia do Okapi por documentos mais curtos
class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    # Calculo do IDF (Inverse Document Frequency)
    def _calc_idf(self, nd):
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[sys.intern(word)] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[sys.intern(word)] = eps

    # Calculo do ctd
    def get_ctd(self, q_freq, b, doc_len, avg_len):
        ctd = q_freq/(1 - b + b*(doc_len)/(avg_len))
        return ctd

    # Avaliar a pontuacao de todos os documentos na base
    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)

        # Funcionamento de term_freqs e term_docs
        # Ex: term_freqs['termo'] = [10, 5, 4, 15] => frequências do termo (TF > 0)
        #     term_docs['termo'] = [5, 20, 40, 55] => termo ocorre nos docs 5, 20, 40, 55 
        for q in query:
            if q not in self.term_freqs:
                continue
            q_tf = [0]*self.corpus_size 
            for docn, tf in zip(self.term_docs[q], self.term_freqs[q]):
                q_tf[docn] = tf
            ctd = q_tf / (1 - self.b + self.b * (doc_len) / (self.avgdl))
            score += (self.idf.get(q, 0)) * ((ctd + 0.5) * (self.k1 + 1) / ((ctd + 0.5) + self.k1))

        return score


    def get_top_n(self, query, documents, n=5,
                  improve_similarity=False, raw_query=None, past_queries=[],
                  retrieved_docs=[], names=[], cut=DEFAULT_CUT, delta=DEFAULT_DELTA,
                  peso_pouco_relevantes=DEFAULT_PESO_POUCO_RELEVANTES):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus"

        scores = self.get_scores(query)

        if np.isclose(np.max(scores), np.min(scores), atol=1e-5):
            score_ref = 1.0 if np.max(scores) > 1e-6 else 0.0
            scores_normalized = np.array([score_ref for i in range(len(scores))])
        else:
            scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        scores_final = np.copy(scores_normalized)
        if (improve_similarity and len(past_queries) > 0):
            lambdas = lambda_calc(all_queries=past_queries, retrieved_docs=retrieved_docs,
                                        query=raw_query, cut=cut, delta=delta,
                                        peso_pouco_relevantes=peso_pouco_relevantes)
            scores_final = lambda_update(scores=scores_normalized, lambdas=lambdas, names=names)

        top_n = np.argpartition(scores_final, -n)[::-1][:n]
        top_n = top_n[np.argsort(scores_final[top_n])[::-1]]

        return [documents[i] for i in top_n], [scores[i] for i in top_n], [scores_normalized[i] for i in top_n], [scores_final[i] for i in top_n]
