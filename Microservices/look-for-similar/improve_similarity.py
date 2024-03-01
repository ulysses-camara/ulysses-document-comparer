import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def lambda_update(scores, lambdas, names):
    """
    Updates bm25 scores using the lambdas values
    """
    result = np.copy(scores)
    for i, name in enumerate(names):
        name = name.strip()
        if name in lambdas.keys():
            result[i] += lambdas[name]
    return result


def lambda_calc(all_queries, retrieved_docs, query, cut, delta, peso_pouco_relevantes):
    """
    Searches for similar queries; returns dictionary
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_queries + [query])
    vsm_2 = vectorizer.transform(all_queries)
    vsm_1 = vectorizer.transform([query])
    similarities = cosine_similarity(vsm_1, vsm_2).tolist()[0]

    doc_sim = [(retrieved_docs[j], similarities[j]) for j in range(len(similarities)) if similarities[j] > cut]

    dic = {}
    for tuple in doc_sim:
        sim = tuple[1]
        for doc in tuple[0]:
            (name, score, score_norm, relevance) = doc
            if relevance == 'i':
                rel = -1
            elif relevance == 'pr':
                rel = peso_pouco_relevantes
            else:
                rel = 1
            if name in dic:
                dic[name] += score_norm * sim * rel
            else:
                dic[name] = score_norm * sim * rel  # calculando a soma do produto sim*score*rel

    for key in dic:
        dic[key] = np.log(dic[key] + 1) * delta
    return dic
