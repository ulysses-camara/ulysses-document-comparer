import os
from ast import literal_eval
import json
import requests
import csv
from io import StringIO
from flask import Flask, request, jsonify, Response
import psycopg2
from numpy import transpose, array

from bm25 import BM25L
from preprocessing import preprocess

import base64
from cryptography.fernet import Fernet

SELECT_PROPS_LEGS = "SELECT code, name, txt_ementa, text_preprocessed FROM proposicoes_legislativas;"
SELECT_CONSULTAS = "SELECT code, name, text, text_preprocessed FROM consultas_legislativas where length(name) > 1;"

#Nao insere todas as colunas, exceto aquelas necessaria para essa aplicacao atualmente (load_props_legs(), antiga load_corpus()).
INSERT_PROP_LEG = "INSERT INTO proposicoes_legislativas (code, name, txt_ementa, text, text_preprocessed) VALUES ('{}','{}','{}','{}','{}');"

SELECT_ROOT_BY_PROPOSICAO = "SELECT cod_proposicao_raiz FROM arvore_proposicoes WHERE cod_proposicao = {}"
SELECT_AVORE_BY_RAIZ = "SELECT * FROM arvore_proposicoes WHERE cod_proposicao_raiz IN {}"

SELECT_ARVORE_BY_PROPOSICAO = "SELECT * FROM arvore_proposicoes WHERE cod_proposicao = {}"

INSERT_DATA_ST = "INSERT INTO consultas_legislativas (name, text, text_preprocessed) VALUES ('{}','{}','{}');"
INSERT_DATA_ST_teste = "INSERT INTO consultas_legislativas (name, text, text_preprocessed) VALUES "
teste2 = "('{}','{}','{}')"

session = requests.Session()
session.trust_env = False
## retrocar ulyssesdb de localhost #####
crypt_key = os.getenv('CRYPT_KEY_SOLIC_TRAB', default='')
url_expand_query = os.getenv('URL_EXPAND_QUERY', default='http://expand-query:5003')
db_connection = os.getenv('ULYSSES_DB_CONNECTION', default='host=localhost dbname=admin user=admin password=admin port=5432')
app = Flask(__name__)

def load_props_legs(db_connection):
    conn = psycopg2.connect(db_connection)
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(SELECT_PROPS_LEGS)
                (codes, names, ementas, tokenized_corpus) = [], [], [], []
                for code, name, ementa, text_preprocessed in cursor:
                    codes.append(code)
                    names.append(name)
                    ementas.append(ementa)
                    tokenized_corpus.append(literal_eval(text_preprocessed))
                (codes, names, ementas) = (array(codes), array(names), array(ementas))
    finally:
        conn.close()
    print("Loaded", len(names), "documents")
    return (codes, names, ementas, tokenized_corpus)

def load_solicitacoes(db_connection):
    crypt_key = os.getenv('CRYPT_KEY_SOLIC_TRAB', default='')
    net = Fernet(crypt_key.encode()) if crypt_key else None

    conn = psycopg2.connect(db_connection)
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(SELECT_CONSULTAS)
                (codes, names, texts, tokenized_sts) = [], [], [], []
                for code, name, text, tokenized_st in cursor:
                    text_plain = net.decrypt(base64.b64decode(text)).decode() if net else text
                    tokenized_st_plain = net.decrypt(base64.b64decode(tokenized_st)).decode() if net else tokenized_st
                    codes.append(code)
                    names.append(name)
                    texts.append(text_plain)
                    tokenized_sts.append(literal_eval(tokenized_st_plain))
                (codes, names, texts) = (array(codes), array(names), array(texts))
    finally:
        conn.close()
    print("Loaded", len(names), "Solicitações de Trabalho")
    return (codes, names, texts, tokenized_sts)


# Loading data
print("Loading corpus...")
(codes, names, ementas, tokenized_corpus) = load_props_legs(db_connection)
(codes_sts, names_sts, texto_sts, tokenized_sts) = load_solicitacoes(db_connection)

# Loading model with dataset
model = BM25L(tokenized_corpus)
model_st = BM25L(tokenized_sts)
print("Modelos carregados com sucesso")

def getPastFeedback(db_connection):
    conn = psycopg2.connect(db_connection)
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT query, user_feedback FROM feedback;")
                (queries, feedbacks) = [], []
                for query, feedback in cursor:
                    queries.append(query)
                    feedbacks.append(literal_eval(feedback))
    finally:
        conn.close()
    scores = []
    all_queries = []
    for entry, q in zip(feedbacks, queries):
        scores.append([[i["id"], float(i["score"]), float(i["score_normalized"])] for i in entry if i["class"]!='i'])
        all_queries.append(q)
    return all_queries, scores

def retrieveDocuments(query, n, raw_query, improve_similarity, past_queries, past_scores):
    indexes = list(range(len(codes)))
    slice_indexes, scores, scores_normalized, scores_final = model.get_top_n(query, indexes, n=n,
            improve_similarity=improve_similarity, raw_query= raw_query, past_queries=past_queries,
            retrieved_docs=past_scores, names=names)
    selected_codes = codes[slice_indexes]
    selected_ementas = ementas[slice_indexes]
    selected_names = names[slice_indexes]
    return selected_codes, selected_ementas, selected_names, scores, scores_normalized, scores_final

def retrieveSTs(query, n, raw_query, improve_similarity, past_queries, past_scores):
    indexes = list(range(len(names_sts)))
    slice_indexes, scores, scores_normalized, scores_final = model_st.get_top_n(query, indexes, n=n,
            improve_similarity=improve_similarity, raw_query= raw_query, past_queries=past_queries,
            retrieved_docs=past_scores, names=names_sts)
    selected_sts = texto_sts[slice_indexes]
    selected_names = names_sts[slice_indexes]
    selected_codes = codes_sts[slice_indexes]
    return selected_codes, selected_names, selected_sts, scores, scores_normalized, scores_final


def getRelationsFromTree(retrieved_doc, db_connection):
    conn = psycopg2.connect(db_connection)
    try:
        with conn:
            with conn.cursor() as cursor:
                '''
                cursor.execute(SELECT_ROOT_BY_PROPOSICAO.format(retrieved_doc))
                roots = cursor.fetchall()

                # Considerando que documento seja uma raiz
                if (len(roots) == 0):
                    roots.append((retrieved_doc,))

                roots = [str(i[0]) for i in roots]
                cursor.execute(SELECT_AVORE_BY_RAIZ.format("(%s)" % ",".join(roots)))
                results = cursor.fetchall()
                '''
                #nao existe mais coluna raiz, logo procuraremos direto pela proposicao
                cursor.execute(SELECT_ARVORE_BY_PROPOSICAO.format(retrieved_doc))
                results = cursor.fetchall()
                
                '''
                results = list(map(lambda x: {"numero_sequencia": x[0], "nivel": x[1], "cod_proposicao": x[2],
                                              "cod_proposicao_referenciada": x[3], "cod_proposicao_raiz": x[4],
                                              "tipo_referencia": x[5]}, results))
                '''
                #Adequando resultado as novas colunas do db
                results = list(map(lambda x: {"code": x[0], "numero_sequencia": x[1], "cod_proposicao": x[2],
                                              "cod_proposicao_referenciada": x[3],
                                              "tipo_referencia": x[4]}, results))
                return results
    finally:
        conn.close()


@app.route('/', methods=["POST"])
def lookForSimilar():
    args = request.json
    try:
        query = args["text"]
    except:
        return ""
    try:
        k_prop = args["num_proposicoes"]
    except:
        k_prop = 20
    try:
        k_st = args["num_solicitacoes"]
    except:
        k_st = 20
    try:
        query_expansion = bool(args["expansao"]) #bool retorna True se for uma string o parametro; Cuidado na hora de fazer a request.
    except:
        query_expansion = True
    try:
        improve_similarity = bool(args["improve-similarity"])
    except:
        improve_similarity = True
        
    try:
        use_relations_tree = bool(args["use_relations_tree"])
    except:
        use_relations_tree = False 

    k_prop = min(k_prop, len(codes))
    k_st = min(k_st, len(names_sts))

    if (query_expansion):
        resp = session.post(url_expand_query, json={"query": query})
        if (resp.status_code == 200):
            query = json.loads(resp.content)["query"]
    preprocessed_query = preprocess(query)

    # Recuperar feedbacks de relevância passados quando necessário
    if improve_similarity:
        past_queries, past_scores = getPastFeedback(db_connection)
    else:
        past_queries, past_scores = None, None

    # Recuperando das solicitações de trabalho
    selected_codes_sts, selected_names_sts, selected_sts, scores_sts, scores_sts_normalized, scores_sts_final = retrieveSTs(preprocessed_query, k_st,
                                                                improve_similarity=improve_similarity, raw_query=query,
                                                                past_queries=past_queries, past_scores=past_scores)
    resp_results_sts = list()
    for i  in range(k_st):
        resp_results_sts.append({"id": int(selected_codes_sts[i]), "name": selected_names_sts[i], "texto": selected_sts[i].strip(),
                    "score": scores_sts[i], "score_normalized": scores_sts_normalized[i],
                    "score_final": scores_sts_final[i], "tipo": "ST"})


    # Recuperando do corpus das proposições
    selected_codes, selected_ementas, selected_names, scores, scores_normalized, scores_final = retrieveDocuments(preprocessed_query, k_prop,
                                                                improve_similarity=improve_similarity, raw_query=query,
                                                                past_queries=past_queries, past_scores=past_scores)
    resp_results = list()
    if (use_relations_tree):
        for i in range(k_prop):
            # Propostas relacionadas pela árvore de proposições
            relations_tree = getRelationsFromTree(selected_codes[i], db_connection)
            resp_results.append({"id": int(selected_codes[i]), "name": selected_names[i],
                        "texto": selected_ementas[i].strip(), "score": scores[i],
                        "score_normalized": scores_normalized[i],
                        "score_final": scores_final[i], "tipo": "PR", "arvore": relations_tree})
    else:
        for i in range(k_prop):
            resp_results.append({"id": int(selected_codes[i]), "name": selected_names[i],
                        "texto": selected_ementas[i].strip(), "score": scores[i],
                        "score_normalized": scores_normalized[i],
                        "score_final": scores_final[i], "tipo": "PR"})

    response = {"proposicoes": resp_results, "solicitacoes": resp_results_sts, "actual_query": query}
    return jsonify(response)


@app.route('/insert', methods=["POST"])
def insertDocs():
    content = request.data.decode("utf-8")
    io = StringIO(content)
    reader = csv.reader(io, delimiter=',')

    data = [row for row in reader]
    columns = data[0]
    data = data[1:]

    try:
        idx_text = columns.index("text")
        idx_ementa = columns.index("txt_ementa")
        idx_code = columns.index("code")
        idx_name = columns.index("name")

        data = transpose( data )

        text = data[idx_text]
        txt_ementa = data[idx_ementa]
        code = data[idx_code]
        name = data[idx_name]
        text_preprocessed = ['{' + ','.join(['"'+str(entry)+'"' for entry in preprocess(txt)]) + '}' for txt in text]

        data_insert = transpose( [code, name, txt_ementa, text, text_preprocessed] )

        conn = psycopg2.connect(db_connection)
        try:
            with conn:
                with conn.cursor() as cursor:
                    for d in data_insert:
                        cursor.execute(INSERT_PROP_LEG.format(*d)) #Colunas faltantes podem ser nulas: nao deve haver problemas
        finally:
            conn.close()

        # Reloading model
        print("RELOADING...")
        global model, codes, names, ementas, tokenized_corpus
        (codes, names, ementas, tokenized_corpus) = load_props_legs(db_connection)
        model = BM25L(tokenized_corpus)
        print("RELOAD DONE")
    except:
        return Response(status=500)

    return Response(status=201)

@app.route('/insert-forced-sts', methods=["POST"]) #inserir planilha sts sem a primeira virgula, antes de name
def insertForcesDocs():
    print("FORCIBLY INSERTING DB")
    content = request.data.decode("utf-8")
    io = StringIO(content)
    reader = csv.reader(io, delimiter=',')

    data = [row for row in reader]
    columns = data[0]
    data = data[1:]

    try:
        idx_name = columns.index("name")
        idx_text = columns.index("text")
        idx_text_preprocessed = columns.index("text_preprocessed")

        data = transpose( data )

        name = data[idx_name]
        text = data[idx_text]
        text_preprocessed = data[idx_text_preprocessed]
        text_preprocessed = [word.replace("'",'"') for word in text_preprocessed]


        data_insert = transpose([name, text, text_preprocessed])

        command = INSERT_DATA_ST_teste
        flag = False
        for data in data_insert:
            if (flag):
                command = command + ','
            command = command + '(' + "'" + data[0] + "'" + ',' + "'" + data[1] + "'" + ',' + "'" + data[2] + "'" + ')'
            if (flag == False):
                flag = True

        command = command + ';'
        '''
        i = 0
        print(len(data_insert))
        with connection.cursor() as cursor:
            for d in data_insert:
                pendingCommit = True
                cursor.execute(INSERT_DATA_ST.format(*d))
                i = i + 1
                if(i == 99):
                    connection.commit()
                    i = 0
                    pendingCommit = False
                #print(i)
            print("AQUI 9")
            if(pendingCommit):
                connection.commit()
            print("AQUI 10")
        '''

        conn = psycopg2.connect(db_connection)
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(command)
        finally:
            conn.close()

        # Reloading model
        print("RELOADING...")
        global codes_sts, names_sts, texto_sts, tokenized_sts, model_st
        (codes_sts, names_sts, texto_sts, tokenized_sts) = load_solicitacoes(db_connection)
        model_st = BM25L(text_preprocessed)
        print("RELOAD DONE")
    except:
        return Response(status=500)

    return Response(status=201)



if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5000)
