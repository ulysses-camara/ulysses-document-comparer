"""
uncertain_metrics.py  Calcula métricas com incerteza.
"""

import numpy as np
from .metrics import Metrics

# Instancia a classe de métricas
metrics = Metrics()


"""
Calcula precision_k com incerteza
"""
def precision_k(feedbacks: str, k: int):

    if not feedbacks:
        return (None, None)
    
    # cálculo sem incerteza
    qtd_hifen = feedbacks.count("-")
    if qtd_hifen == 0:
        return (metrics.precision_k(feedbacks, k), 0.0)

    # abaixo, cálculo com incerteza
    lst_precision_k = []

    # primeiro calcula substituindo todos os "-" por "i"
    lst_precision_k.append(metrics.precision_k(feedbacks.replace("-", "i"), k))
    # depois calcula substituindo todos os "-" por "r"
    lst_precision_k.append(metrics.precision_k(feedbacks.replace("-", "r"), k))

    mean_precision_k = np.mean(lst_precision_k)

    return (mean_precision_k, np.max(lst_precision_k) - mean_precision_k)


"""
Calcula recall_k com incerteza
"""
def recall_k(feedbacks: str, n_relevant_docs: float, k: int):

    if not feedbacks:
        return (None, None)
    if n_relevant_docs == 0.0:
        return (0.0, 0.0)

    # cálculo sem incerteza
    qtd_hifen = feedbacks.count("-")
    if qtd_hifen == 0:
        return (metrics.recall_k(feedbacks, n_relevant_docs, k), 0.0)

    lst_recall_k = []

    # primeiro calcula substituindo todos os "-" por "i"
    lst_recall_k.append(metrics.recall_k(feedbacks.replace("-", "i"), n_relevant_docs, k))
    # depois calcula substituindo todos os "-" por "r"
    lst_recall_k.append(metrics.recall_k(feedbacks.replace("-", "r"), (n_relevant_docs + qtd_hifen), k))

    mean_recall_k = np.mean(lst_recall_k)

    return (mean_recall_k, np.max(lst_recall_k) - mean_recall_k)

"""
Calcula average_precision com incerteza
"""
def average_precision(feedbacks: str, n_relevant_docs: float):

    if not feedbacks:
        return (None, None)
    if n_relevant_docs == 0.0:
        return (0.0, 0.0)

    # cálculo sem incerteza
    qtd_hifen = feedbacks.count("-")
    if qtd_hifen == 0:
        return (metrics.average_precision(feedbacks, n_relevant_docs), 0.0)

    lst_avg_precision = []

    # primeiro calcula substituindo todos os "-" por "i"
    lst_avg_precision.append(metrics.average_precision(feedbacks.replace("-", "i"), n_relevant_docs))
    # depois calcula substituindo todos os "-" por "r"
    lst_avg_precision.append(metrics.average_precision(feedbacks.replace("-", "r"), (n_relevant_docs + qtd_hifen)))

    mean_avg_precision = np.mean(lst_avg_precision)

    return (mean_avg_precision, np.max(lst_avg_precision) - mean_avg_precision)


"""
Calcula r_precision com incerteza
"""
def r_precision(feedbacks: str, n_relevant_docs: float):

    if not feedbacks:
        return (None, None)
    if n_relevant_docs == 0.0:
        return (0.0, 0.0)

    # cálculo sem incerteza
    qtd_hifen = feedbacks.count("-")
    if qtd_hifen == 0:
        return (metrics.r_precision(feedbacks, n_relevant_docs), 0.0)

    lst_r_precision = []

    # primeiro calcula substituindo todos os "-" por "i"
    lst_r_precision.append(metrics.r_precision(feedbacks.replace("-", "i"), n_relevant_docs))
    # depois calcula substituindo todos os "-" por "r"
    lst_r_precision.append(metrics.r_precision(feedbacks.replace("-", "r"), (n_relevant_docs + qtd_hifen)))

    mean_r_precision = np.mean(lst_r_precision)

    return (mean_r_precision, np.max(lst_r_precision) - mean_r_precision)


"""
Calcula reciprocal_rank com incerteza
"""
def reciprocal_rank(feedbacks: str):

    if (not feedbacks):
        return (None, None)

    # cálculo sem incerteza
    qtd_hifen = feedbacks.count("-")
    if qtd_hifen == 0:
        return (metrics.reciprocal_rank(feedbacks), 0.0)

    lst_reciprocal_rank = []

    # primeiro calcula substituindo todos os "-" por "i"
    lst_reciprocal_rank.append(metrics.reciprocal_rank(feedbacks.replace("-", "i")))
    # depois calcula substituindo todos os "-" por "r"
    lst_reciprocal_rank.append(metrics.reciprocal_rank(feedbacks.replace("-", "r")))

    arr_reciprocal_rank = np.array(lst_reciprocal_rank)

    return (arr_reciprocal_rank.mean(), arr_reciprocal_rank.max() - arr_reciprocal_rank.mean())
