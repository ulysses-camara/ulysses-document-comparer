"""
metrics_utils.py  Faz o "meio de campo" entre os feedbacks U3 gravados na base e o metrics.py

>> Campo 'user_feedback' da base de dados: lista de dicionários, um dicionário corresponde a um feedback individual.

Exemplo de um feedback de proposição:
{"id": "PL 4251/2019", "class": "i", "score": "25.567451558644", "score_normalized": "1.0", "score_final": "1.0417292694402227", "tipo": "PR"}

Exemplo de um feedback de ST:
{"id": "12047/2015", "class": "i", "score": "41.17354103357339", "score_normalized": "0.3885914187319897", "score_final": "0.4303206881722125", "tipo": "ST"}

>> Campo 'extra_results' da base de dados: lista de strings, cada string cooresponde a um documento (proposição ou ST).

Exemplo de uma ocorrência de 'extra_results':
["PL 7485/2006", "PL 6480/2009", "21839/2019", "17188/2019", "15024/2019", "17193/2011"]
"""

from typing import List, Dict, Tuple


def feedbacks_to_strings(feedbacks: List[Dict]):

    # Recebe uma lista de feedbacks vindos da base de dados e converte em uma string. Separa os
    # feedbacks de proposições e solicitações de trabalho.

    feedbacks_pr = ''
    feedbacks_st = ''

    for feedback in feedbacks:
        
        fb_class = feedback.get('class', None) 
        fb_tipo = feedback.get('tipo', None)

        if not fb_class:
            fb_class = '-' # sem classificação - hífen

        if not fb_tipo:
            continue # despreza o feedback sem tipo

        switcher = fb_class.lower() + fb_tipo.upper() # classificação deve ser caixa baixa e tipo deve ser caixa alta
        if switcher == 'rPR':
            feedbacks_pr += 'r'
        elif switcher == 'prPR':
            feedbacks_pr += 'p'
        elif switcher == 'iPR':
            feedbacks_pr += 'i'
        elif switcher == '-PR':
            feedbacks_pr += '-'
        elif switcher == 'rST':
            feedbacks_st += 'r'
        elif switcher == 'prST':
            feedbacks_st += 'p'
        elif switcher == 'iST':
            feedbacks_st += 'i'
        elif switcher == '-ST':
            feedbacks_st += '-'
        else:
            raise ValueError('Classificação ou Tipo errado em feedback.')

    return feedbacks_pr if feedbacks_pr != '' else None, feedbacks_st if feedbacks_st != '' else None


def extra_results_to_numbers(extra_results: List[str]):

    # Converte os extra_results em quantidades. Separa os extra_results de proposições e de STs.
    # Trata os extra_results fora do padrão.

    def valida_tipo(doc: str):

        import re

        doc = doc.upper().strip() # tem q ser caixa alta e sem espaços no início e final

        if doc == '':
            # documento inválido
            return "INV"
        
        # verifica se o documento se encaixa em um dos padrões aceitáveis:
        # <Sigla> <Número>/<Ano> para proposições, ou;
        # <Número>/<Ano> para solicitações
        doc = re.sub(r' */ *', '/', doc) # retira espaços (se houver) antes ou depois da barra
        padroes = r'([A-Z]+ +[0-9]+/([0-9]{2}|[0-9]{4})$)|([0-9]+/([0-9]{2}|[0-9]{4})$)'
        if not re.match(padroes, doc):
            return "INV"
        
        # procura pelo espaço obrigatório entre a sigla e o número, se for proposição
        pos = doc.find(' ')

        if pos == -1:
            # Não encontrou; é uma ST
            return "ST"

        # isola a Sigla
        sigla = doc[:pos]

        # analisa a Sigla
        if sigla in ["ST", "SOL"]:
            # é uma ST
            return "ST"
        #
        if sigla in ["LEI", "MPV"]:
            # documento inválido
            return "INV"
        #
        return "PR"

    qtd_extra_pr = 0
    qtd_extra_st = 0

    for doc in extra_results:

        # doc pode ser proposição ou ST

        if valida_tipo(doc) == "PR":
            qtd_extra_pr += 1
        elif valida_tipo(doc) == "ST":
            qtd_extra_st += 1
        else:
            continue # documento inválido para efeito de contagem de relevantes - Lei, MPV etc.

    return qtd_extra_pr, qtd_extra_st


def run_metrics(db_connection, period: Tuple[str]):

    '''
    calcula_metricas.py  Calcula as métricas U3 para um determinado período de feedbacks
    '''

    import pandas as pd

    from .uncertain_metrics import precision_k, recall_k, average_precision, r_precision, reciprocal_rank
    from datetime import date
    from ..config_data import tb_feedback

    # Selects
    SELECT_FEEDBACKS = f"SELECT id, user_feedback, extra_results FROM {tb_feedback} "
    if period:
        SELECT_FEEDBACKS += f"WHERE convert(date, dat_conclusao_st) between '{period[0]}' and '{period[1]}'"
    else:
        SELECT_FEEDBACKS += f"WHERE dat_conclusao_st is not null" # pega todos os feedbacks do Sisconle
    #
    # SELECT_DATE = 'SELECT FORMAT(GETDATE(), \'yyyy-MM-dd HH:mm\')'

    # Tamanho da janela de documentos retornados
    TAM_JANELA_DOCS = 12

    ###################
    # Carga dos dados #
    ###################

    # Feedbacks
    queries_fb = pd.read_sql(SELECT_FEEDBACKS, db_connection)

    # DB date
    # db_date = db_connection.execute(SELECT_DATE).scalar()

    # Fecha a conexão 
    db_connection.close()

    # Verifica se há dados de feedback
    if queries_fb.shape[0] == 0:
        # Não recuperou nenhuma linha de feedback
        results = {'Dat_execucao': str(date.today()), 'Periodo': {'Dat_inicio': period[0], 'Dat_fim': period[1]}, 'Mensagem': 'Não há feedbacks para o período selecionado.'}
        return results

    ################################################################
    # Converte feedbacks em strings e extra_results em quantidades #
    ################################################################

    queries_fb['feedbacks_pr'], queries_fb['feedbacks_st'] = zip(*queries_fb['user_feedback'].fillna('[]').apply(eval).apply(feedbacks_to_strings))
    queries_fb['qtd_extra_pr'], queries_fb['qtd_extra_st'] = zip(*queries_fb['extra_results'].fillna('[]').apply(eval).apply(extra_results_to_numbers))

    ##############################################
    # Faz o padding para janela de 12 documentos #
    ##############################################

    queries_fb['feedbacks_pr'] = queries_fb['feedbacks_pr'].str.ljust(TAM_JANELA_DOCS, '-')
    queries_fb['feedbacks_st'] = queries_fb['feedbacks_st'].str.ljust(TAM_JANELA_DOCS, '-')

    ####################################################
    # Calcula o total de documentos relevantes (float) #
    ####################################################

    queries_fb['total_relevantes_pr'] = queries_fb['qtd_extra_pr'] + queries_fb['feedbacks_pr'].str.count('r') + 0.2*queries_fb['feedbacks_pr'].str.count('p')
    queries_fb['total_relevantes_st'] = queries_fb['qtd_extra_st'] + queries_fb['feedbacks_st'].str.count('r') + 0.2*queries_fb['feedbacks_st'].str.count('p')

    queries_fb['total_relevantes_pr'].fillna(0.0, inplace=True)
    queries_fb['total_relevantes_st'].fillna(0.0, inplace=True)

    #########################################################################################
    # Elimina linhas onde não há informação de feedback nem para proposições E nem para STs #
    #########################################################################################

    queries_fb.dropna(axis='index', how='all', subset=['feedbacks_pr', 'feedbacks_st'], inplace=True)

    ######################################################################
    # Verifica se ainda há dados no DataFrame, após os tratamentos acima #
    ######################################################################

    if queries_fb.shape[0] == 0:
        # Não há feedbacks válidos
        results = {'Dat_execucao': str(date.today()), 'Periodo': {'Dat_inicio': period[0], 'Dat_fim': period[1]}, 'Mensagem': 'Não há feedbacks válidos para o período selecionado.'}
        return results

    #######################
    # CALCULA AS MÉTRICAS #
    #######################

    #############
    # Precision #
    #############

    for i in range(TAM_JANELA_DOCS):
        queries_fb[f'p@{i+1}_pr'] = queries_fb.apply(lambda x: precision_k(x['feedbacks_pr'], i+1), axis=1)
        queries_fb[f'p@{i+1}_st'] = queries_fb.apply(lambda x: precision_k(x['feedbacks_st'], i+1), axis=1)

    ##########
    # Recall #
    ##########

    for i in range(TAM_JANELA_DOCS):
        queries_fb[f'r@{i+1}_pr'] = queries_fb.apply(lambda x: recall_k(x['feedbacks_pr'], x['total_relevantes_pr'], i+1), axis=1)
        queries_fb[f'r@{i+1}_st'] = queries_fb.apply(lambda x: recall_k(x['feedbacks_st'], x['total_relevantes_st'], i+1), axis=1)

    #####################
    # Average Precision #
    #####################

    queries_fb['avg_precision_pr'] = queries_fb.apply(lambda x: average_precision(x['feedbacks_pr'], x['total_relevantes_pr']), axis=1)
    queries_fb['avg_precision_st'] = queries_fb.apply(lambda x: average_precision(x['feedbacks_st'], x['total_relevantes_st']), axis=1)

    ###############
    # R-Precision #
    ###############

    queries_fb['r_precision_pr'] = queries_fb.apply(lambda x: r_precision(x['feedbacks_pr'], x['total_relevantes_pr']), axis=1)
    queries_fb['r_precision_st'] = queries_fb.apply(lambda x: r_precision(x['feedbacks_st'], x['total_relevantes_st']), axis=1)

    ###################
    # Reciprocal Rank #
    ###################

    queries_fb['rec_rank_pr'] = queries_fb.apply(lambda x: reciprocal_rank(x['feedbacks_pr']), axis=1)
    queries_fb['rec_rank_st'] = queries_fb.apply(lambda x: reciprocal_rank(x['feedbacks_st']), axis=1)

    #################################################################
    # Calcula as médias das métricas, considerando todas as queries #
    #################################################################

    # calcula as médias das tuplas (métrica, incerteza)
    def calcula_media(serie: pd.Series):

        import numpy as np

        lst_metricas = []
        lst_incertezas = []

        for row in serie:
            if not pd.isna(row[0]):
                lst_metricas.append(row[0])
            if not pd.isna(row[1]):
                lst_incertezas.append(row[1])
        
        assert(len(lst_metricas) == len(lst_incertezas))

        if len(lst_metricas) == 0:
            return {'valor': None, 'incerteza': None}
        
        return {'valor': np.mean(lst_metricas), 'incerteza': np.mean(lst_incertezas)}

        # TODO: revisar média das incertezas
        # 0.80 +/- 0.04    => de 0.76 até 0.84
        # 0.90 +/- 0.08    => de 0.82 até 0.98


    #### Proposições ####
    mean_pr = {}
    # Precision
    for i in range(TAM_JANELA_DOCS):
        metrica = calcula_media(queries_fb[f'p@{i+1}_pr'])
        if metrica['valor'] is not None:
            mean_pr.update({f'mean_p@{i+1}': metrica})
    # Recall
    for i in range(TAM_JANELA_DOCS):
        metrica = calcula_media(queries_fb[f'r@{i+1}_pr'])
        if metrica['valor'] is not None:
            mean_pr.update({f'mean_r@{i+1}': metrica})
    # Mean Average Precision
    metrica = calcula_media(queries_fb['avg_precision_pr'])
    if metrica['valor'] is not None:
        mean_pr.update({'MAP': metrica})
    # Average R-Precision
    metrica = calcula_media(queries_fb['r_precision_pr'])
    if metrica['valor'] is not None:
        mean_pr.update({'ARP': metrica})
    # Mean Reciprocal Rank
    metrica = calcula_media(queries_fb['rec_rank_pr'])
    if metrica['valor'] is not None:
        mean_pr.update({'MRR': metrica})

    #### Solicitações ####
    mean_st = {}
    # Precision
    for i in range(TAM_JANELA_DOCS):
        metrica = calcula_media(queries_fb[f'p@{i+1}_st'])
        if metrica['valor'] is not None:
            mean_st.update({f'mean_p@{i+1}': metrica})
    # Recall
    for i in range(TAM_JANELA_DOCS):
        metrica = calcula_media(queries_fb[f'r@{i+1}_st'])
        if metrica['valor'] is not None:
            mean_st.update({f'mean_r@{i+1}': metrica})
    # Mean Average Precision
    metrica = calcula_media(queries_fb['avg_precision_st'])
    if metrica['valor'] is not None:
        mean_st.update({'MAP': metrica})
    # Average R-Precision
    metrica = calcula_media(queries_fb['r_precision_st'])
    if metrica['valor'] is not None:
        mean_st.update({'ARP': metrica})
    # Mean Reciprocal Rank
    metrica = calcula_media(queries_fb['rec_rank_st'])
    if metrica['valor'] is not None:
        mean_st.update({'MRR': metrica})

    ##########################
    # Retorno dos Resultados #
    ##########################

    # # tratamento de valores NaN nos dicionários de retorno
    # def replace_na(dict_: dict):
    #     for key, value in dict_.items():
    #         if type(value) is dict:
    #             replace_na(value)
    #         elif pd.isna(value):
    #             dict_[key] = None
    # #
    # replace_na(mean_pr)
    # replace_na(mean_st)

    results = {'Dat_execucao': str(date.today()), 'Periodo': {'Dat_inicio': period[0], 'Dat_fim': period[1]}}
    if mean_pr != {}:
        results['Proposicoes'] = mean_pr
    if mean_st != {}:
        results['Solicitacoes'] = mean_st

    return results
