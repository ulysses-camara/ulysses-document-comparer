"""
metrics.py  Calcula métricas na recuperação de documentos, conforme PDF da USP.

Classificações permitidas:
'r': relevante; 'p': pouco relevante; 'i': irrelevante; '-': sem classificação
"""

class Metrics:

    # Classificações possíveis
    RELEVANTE = 'r'
    POUCO_RELEVANTE = 'p'
    IRRELEVANTE = 'i'
    SEM_CLASSIFICACAO = '-'

    # Pesos das classificações
    PESO_RELEVANTE = 1.0
    PESO_POUCO_RELEVANTE = 0.2
    PESO_IRRELEVANTE = 0.0
    PESO_SEM_CLASSIFICACAO = 0.0

    # Tamanho default da janela de documentos retornados
    QTD_DOCUMENTOS_RETORNADOS = 12


    def __init__(self, peso_relevante=PESO_RELEVANTE, peso_pouco_relevante=PESO_POUCO_RELEVANTE, peso_irrelevante=PESO_IRRELEVANTE, 
                 peso_sem_classificacao=PESO_SEM_CLASSIFICACAO, qtd_documentos_retornados=QTD_DOCUMENTOS_RETORNADOS):
        self.dict_relevance = {self.RELEVANTE: peso_relevante, 
                               self.POUCO_RELEVANTE: peso_pouco_relevante, 
                               self.IRRELEVANTE: peso_irrelevante, 
                               self.SEM_CLASSIFICACAO: peso_sem_classificacao}
        self.qtd_documentos_retornados = qtd_documentos_retornados


    def recall(self, feedbacks: str, n_relevant_docs: float):

        if not feedbacks:
            return None
        if n_relevant_docs == 0.0:
            return 0.0

        points = 0.0
        for feedback in feedbacks:
            points += self.dict_relevance.get(feedback, 0.0)

        return points / min(n_relevant_docs, self.qtd_documentos_retornados)


    def recall_k(self, feedbacks: str, n_relevant_docs: float, k: int):
        return self.recall(feedbacks[:k], n_relevant_docs)


    def precision(self, feedbacks: str):

        if not feedbacks:
            return None

        points = 0.0
        for feedback in feedbacks:
            points += self.dict_relevance.get(feedback, 0.0)

        return points / len(feedbacks)


    def precision_k(self, feedbacks: str, k: int):
        return self.precision(feedbacks[:k])


    def relevance_k(self, feedbacks: str, k: int):
        return self.dict_relevance.get(feedbacks[k], 0.0)


    def average_precision(self, feedbacks: str, n_relevant_docs: float):

        if not feedbacks:
            return None
        if n_relevant_docs == 0.0:
            return 0.0

        sum_ap = 0.0
        for i in range(len(feedbacks)):
            sum_ap += self.precision_k(feedbacks, i+1)*self.relevance_k(feedbacks, i)

        return sum_ap / min(n_relevant_docs, self.qtd_documentos_retornados)


    def r_precision(self, feedbacks: str, n_relevant_docs: float):

        if not feedbacks:
            return None
        if n_relevant_docs == 0.0:
            return 0.0

        n_relevant_docs = min(n_relevant_docs, self.qtd_documentos_retornados) # limita a qtd. docs. relevantes ao tamanho da janela

        points = 0.0
        for feedback in feedbacks[:int(n_relevant_docs)]:
            points += self.dict_relevance.get(feedback, 0.0)

        return points / n_relevant_docs


    def reciprocal_rank(self, feedbacks: str):
        """ Retorna o PRIMEIRO documento relevante ou pouco relevante, ou seja, o que vier primeiro. """

        if not feedbacks:
            return None
        
        pos_r = feedbacks.find(self.RELEVANTE)
        pos_p = feedbacks.find(self.POUCO_RELEVANTE)

        if (pos_r==-1) and (pos_p==-1):
            # não encontrou nem relevante nem pouco relevante
            return 0.0

        pos_r = len(feedbacks) if pos_r==-1 else pos_r
        pos_p = len(feedbacks) if pos_p==-1 else pos_p
        
        if (pos_r < pos_p):
            return self.PESO_RELEVANTE/(pos_r+1)
        else:
            return self.PESO_POUCO_RELEVANTE/(pos_p+1)
