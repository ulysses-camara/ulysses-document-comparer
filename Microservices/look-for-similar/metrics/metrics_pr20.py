#######################################################################################
# Métricas U3 com peso 0.2 para feedbacks de documentos marcados como PoucoRelevantes #
#######################################################################################


def calcula_precision(queries_fb, labels_all, max_k=12):

    precision = []
    n_queries = len(queries_fb)
    # quant_feedback = list(queries_fb["num_doc_feedback"]) # código USP
    
    for i in range(n_queries): # para todas as queries
        dict_prec = {}
        # for j in range(quant_feedback[i]): # para todos os feedbacks individuais da query (j=0 a quant_feedback-1) - código USP
        for j in range(max_k): # para todos os j de 0 a max_k-1 - nossa correção
            quant = 0.0
            for pr in queries_fb["relevantes"][i]: # para todos os documentos relevantes da query
                if pr in labels_all[i][:j+1]: # se o doc. relevante está entre os top-k (k=j+1)
                    quant += 1.0 # incrementa quant. de documentos relevantes retornados
            for pr in queries_fb["pouco_relevantes"][i]: # para todos os documentos relevantes da query
                if pr in labels_all[i][:j+1]: # se o doc. relevante está entre os top-k (k=j+1)
                    quant += 0.2 # incrementa quant. de documentos relevantes retornados
            dict_prec[j+1] = quant/(j+1) # calcula p@k (precision at k, k=j+1)
                
        precision.append(dict_prec)

    # Daqui em diante, calcula as médias p@k para todas as queries

    lista_medias_precision = {}
    for dic_query in precision:
        for i in range(len(dic_query)):
            if i+1 in lista_medias_precision:
                lista_medias_precision[i+1] += dic_query[i+1]
            else:
                lista_medias_precision[i+1] = dic_query[i+1]
        
            if "quant"+str(i+1) in lista_medias_precision:
                lista_medias_precision['quant'+str(i+1)] += 1
            else:
                lista_medias_precision['quant'+str(i+1)] = 1

    for k in range(int(len(lista_medias_precision)/2)):
        lista_medias_precision[k+1] = lista_medias_precision[k+1]/lista_medias_precision['quant'+str(k+1)]

    results_precision = {}
    for n in range(max_k): # nossa correção
        if n+1 in results_precision:
            results_precision[n+1] += lista_medias_precision[n+1]
        else:
            results_precision[n+1] = lista_medias_precision[n+1]

    return results_precision


def calcula_recall(queries_fb, labels_all, max_k=12):

    recall = []
    n_queries = len(queries_fb)

    # quant_feedback = list(queries_fb["num_doc_feedback"]) # código USP
        
    for i in range(n_queries): # para todas as queries
        qt_relevantes = len(queries_fb["relevantes"][i])*1.0 + len(queries_fb["pouco_relevantes"][i])*0.2 + queries_fb["qt_extra"][i] #Criado variável para armazenar o divisor, levando em consideração a variável qt_extra
        qt_relevantes = min(qt_relevantes, max_k) #Limitando o número de relevantes
        dict_recall = {}
        # for j in range(quant_feedback[i]): # para todos os feedbacks individuais da query (j=0 a quant_feedback-1) - código USP
        for j in range(max_k): # para todos os j de 0 a max_k-1 - nossa correção
            quant = 0.0
            for pr in queries_fb["relevantes"][i]: # para todos os documentos relevantes da query
                if pr in labels_all[i][:j+1]: # se o doc. relevante está entre os top-k (k=j+1) 
                    quant += 1.0 # incrementa quant. de documentos relevantes retornados
            for pr in queries_fb["pouco_relevantes"][i]: # para todos os documentos relevantes da query
                if pr in labels_all[i][:j+1]: # se o doc. relevante está entre os top-k (k=j+1) 
                    quant += 0.2 # incrementa quant. de documentos relevantes retornados
            dict_recall[j+1] = quant/qt_relevantes # calcula r@k (recall at k, k=j+1)

        recall.append(dict_recall)

    # Daqui em diante, calcula as médias r@k para todas as queries

    lista_medias_recall = {}
    for dic_query in recall:
        for i in range(len(dic_query)):
            if i+1 in lista_medias_recall:
                lista_medias_recall[i+1] += dic_query[i+1]
            else:
                lista_medias_recall[i+1] = dic_query[i+1]
        
            if "quant"+str(i+1) in lista_medias_recall:
                lista_medias_recall['quant'+str(i+1)] += 1
            else:
                lista_medias_recall['quant'+str(i+1)] = 1

    for k in range(int(len(lista_medias_recall)/2)):
        lista_medias_recall[k+1] = lista_medias_recall[k+1]/lista_medias_recall['quant'+str(k+1)]

    results_recall = {}
    for n in range(max_k): # nossa correção
        if n+1 in results_recall:
            results_recall[n+1] += lista_medias_recall[n+1]
        else:
            results_recall[n+1] = lista_medias_recall[n+1]

    return results_recall


def calcula_lista_avg_precision(queries_fb, labels_all):

    lista_avg_precision = []

    # quant_feedback = list(queries_fb["num_doc_feedback"])

    for i in range(len(labels_all)): # para todas as queries
        dict_map = {}
        num = 0.0
        for j in range(len(labels_all[i])): # para todos os top-n da query i
            if labels_all[i][j] in queries_fb["relevantes"][i]: # se top-j da query i for relevante
                num += 1.0 # incrementa num
                dict_map[num] = j+1 # (j+1)-ésimo documento com score mais elevado é armazenado em dict_map
            elif labels_all[i][j] in queries_fb["pouco_relevantes"][i]: # se top-j da query i for relevante
                num += 0.2 # incrementa num
                dict_map[num] = j+1 # (j+1)-ésimo documento com score mais elevado é armazenado em dict_map
        lista_avg_precision.append(dict_map)

    return lista_avg_precision


def calcula_avg_precision(queries_fb, lista_avg_precision, max_k=12):

    avg_precision = []

    for i in range(len(lista_avg_precision)):
        qt_relevantes = len(queries_fb["relevantes"][i])*1.0 + len(queries_fb["pouco_relevantes"][i])*0.2 + queries_fb["qt_extra"][i] #Criado variável para armazenar o divisor, levando em consideração a variável qt_extra
        qt_relevantes = min(qt_relevantes, max_k) #Limitando o número de relevantes
        q = lista_avg_precision[i]
        soma = 0
        for key in q:
            soma += key/q[key] # calcula somatório de (precision@k x relevância@k)
        if soma == 0:
            avg_precision.append(0.0) # AP = 0.0
        else:
            avg_precision.append((1/qt_relevantes) * soma) # calcula AP (average precision)

    return avg_precision


def calcula_r_precision(queries_fb, labels_all, max_k=12):

    lista_r_precision = []

    for i in range(len(queries_fb)): # para todas as queries
        rel = len(queries_fb["relevantes"][i])*1.0 + len(queries_fb["pouco_relevantes"][i])*0.2 + queries_fb["qt_extra"][i] # qtd. docs. relevantes para a query, levando em consideração a variável qt_extra
        rel = min(rel, max_k) #Limitando o número de relevantes
        quant = 0.0
        for pr in queries_fb["relevantes"][i]: # para todo documento relevante da query i
            # rel = len(queries_fb["relevantes"][i]) + queries_fb["qt_extra"][i] # qtd. docs. relevantes para a query
            if pr in labels_all[i][:int(rel)]: # se doc relevante estiver até a posição <rel>, então incrementa numerador da fração
                quant += 1.0
        for pr in queries_fb["pouco_relevantes"][i]: # para todo documento relevante da query i
            # rel = len(queries_fb["relevantes"][i]) + queries_fb["qt_extra"][i] # qtd. docs. relevantes para a query
            if pr in labels_all[i][:int(rel)]: # se doc relevante estiver até a posição <rel>, então incrementa numerador da fração
                quant += 0.2
         
        lista_r_precision.append(quant/rel) # calcula RP para todas as queries

    return lista_r_precision


def calcula_rr(lista_avg_precision):

    RR_lista = []

    for doc in lista_avg_precision:
        if 1 in doc:
            RR_lista.append(1.0/doc[1]) # calcula RR = posição do primeiro documento relevante a aparecer
        elif 0.2 in doc:
            RR_lista.append(0.2/doc[0.2]) # calcula RR = posição do primeiro documento relevante a aparecer
        else:
            RR_lista.append(0.0)

    return RR_lista
