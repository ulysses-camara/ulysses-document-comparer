import os
from ast import literal_eval
import json
import requests
import csv
from io import StringIO
from flask import Flask, request, jsonify, Response
from numpy import transpose, array
from bm25 import BM25L
from preprocessing import preprocess
import base64
from cryptography.fernet import Fernet
from sqlalchemy import create_engine


def table_name(name):
    return "".join([c if c.isalnum() else "_" for c in name])


session = requests.Session()
session.trust_env = False
crypt_key = os.getenv('CRYPT_KEY_SOLIC_TRAB', default='')
url_expand_query = os.getenv('URL_EXPAND_QUERY', default='http://expand-query:5003')
db_connection = os.getenv('ULYSSES_DB_CONNECTION', default='postgresql+psycopg2://admin:admin@localhost/admin')
tb_corpus = table_name(os.getenv('TB_CORPUS', default='corpus'))
tb_solicitacoes = table_name(os.getenv('TB_SOLICITACOES', default='solicitacoes'))
tb_feedback = table_name(os.getenv('TB_FEEDBACK', default='feedback'))
db_engine = create_engine(db_connection)

SELECT_CORPUS = f"SELECT code, name, txt_ementa, text_preprocessed FROM {tb_corpus};"
SELECT_ST_FIELDS = f"SELECT code, name, text, text_preprocessed FROM {tb_solicitacoes};"
INSERT_DATA_CORPUS = "INSERT INTO " + tb_corpus + " (code, name, txt_ementa, text, text_preprocessed) VALUES ('{}','{}','{}','{}','{}');"
SELECT_ROOT_BY_PROPOSICAO = "SELECT cod_proposicao_raiz FROM arvore_proposicoes WHERE cod_proposicao = {}"
SELECT_ARVORE_BY_RAIZ = "SELECT * FROM arvore_proposicoes WHERE cod_proposicao_raiz IN {}"
SELECT_FEEDBACK = f"SELECT query, user_feedback FROM {tb_feedback};"

INSERT_DATA_ST = "INSERT INTO " + tb_solicitacoes + " (name, text, text_preprocessed) VALUES ('{}','{}','{}');"
INSERT_DATA_ST_teste = "INSERT INTO " + tb_solicitacoes + " (name, text, text_preprocessed) VALUES "

app = Flask(__name__)


def load_corpus():
    with db_engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            cursor.execute(SELECT_CORPUS)
            (codes, names, ementas, tokenized_corpus) = [], [], [], []
            for code, name, ementa, text_preprocessed in cursor:
                codes.append(code)
                names.append(name)
                ementas.append(ementa)
                tokenized_corpus.append(literal_eval(text_preprocessed))
            (codes, names, ementas) = (array(codes), array(names), array(ementas))
    print("Loaded", len(names), "proposições")
    return codes, names, ementas, tokenized_corpus


def load_solicitacoes():
    crypt_key = os.getenv('CRYPT_KEY_SOLIC_TRAB', default='')
    net = Fernet(crypt_key.encode()) if crypt_key else None

    with db_engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            cursor.execute(SELECT_ST_FIELDS)
            (codes, names, texts, tokenized_sts) = [], [], [], []
            for code, name, text, tokenized_st in cursor:
                text_plain = net.decrypt(base64.b64decode(text)).decode() if net else text
                tokenized_st_plain = net.decrypt(base64.b64decode(tokenized_st)).decode() if net else tokenized_st
                codes.append(code)
                names.append(name)
                texts.append(text_plain)
                tokenized_sts.append(literal_eval(tokenized_st_plain))
            (codes, names, texts) = (array(codes), array(names), array(texts))
    print("Loaded", len(names), "solicitações de trabalho")
    return (codes, names, texts, tokenized_sts)


# Loading data
print("Loading corpus...")
(codes, names, ementas, tokenized_corpus) = load_corpus()
(codes_sts, names_sts, texto_sts, tokenized_sts) = load_solicitacoes()

# Loading model with dataset
model = BM25L(tokenized_corpus)
model_st = BM25L(tokenized_sts)
print("Modelos carregados com sucesso")

def getPastFeedback():
    with db_engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            cursor.execute(SELECT_FEEDBACK)
            (queries, feedbacks) = [], []
            for query, feedback in cursor:
                queries.append(query)
                feedbacks.append(literal_eval(feedback))
    scores = []
    all_queries = []
    for entry, q in zip(feedbacks, queries):
        scores.append([[i["id"], float(i["score"]), float(i["score_normalized"]), i["class"]] for i in entry]) #if i["class"]!='i'])
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


def getRelationsFromTree(retrieved_doc):
    with db_engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            cursor.execute(SELECT_ROOT_BY_PROPOSICAO.format(retrieved_doc))
            roots = cursor.fetchall()

            # Considerando que documento seja uma raiz
            if (len(roots) == 0):
                roots.append((retrieved_doc,))

            roots = [str(i[0]) for i in roots]
            cursor.execute(SELECT_ARVORE_BY_RAIZ.format("(%s)" % ",".join(roots)))
            results = cursor.fetchall()
            results = list(map(lambda x: {"numero_sequencia": x[0], "nivel": x[1], "cod_proposicao": x[2],
                                          "cod_proposicao_referenciada": x[3], "cod_proposicao_raiz": x[4],
                                          "tipo_referencia": x[5]}, results))
            return results


@app.route('/', methods=["POST"])
def lookForSimilar(use_relations_tree = False):
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
        query_expansion = int(args["expansao"])
        if (query_expansion == 0):
            query_expansion = False
        else:
            query_expansion = True
    except:
        query_expansion = True
    try:
        improve_similarity = int(args["improve-similarity"])
        if (improve_similarity == 0):
            improve_similarity = False
        else:
            improve_similarity = True
    except:
        improve_similarity = True

    k_prop = min(k_prop, len(codes))
    k_st = min(k_st, len(names_sts))

    if (query_expansion):
        resp = session.post(url_expand_query, json={"query": query})
        if (resp.status_code == 200):
            query = json.loads(resp.content)["query"]
    preprocessed_query = preprocess(query)

    # Recuperar feedbacks de relevância passados quando necessário
    if improve_similarity:
        past_queries, past_scores = getPastFeedback()
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
            relations_tree = getRelationsFromTree(selected_codes[i])
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

        with db_engine.connect() as conn:
            with conn.begin():
                for d in data_insert:
                    conn.execute(INSERT_DATA_CORPUS.format(*d))

        # Reloading model
        print("RELOADING...")
        global model, codes, names, ementas, tokenized_corpus
        (codes, names, ementas, tokenized_corpus) = load_corpus()
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

        with db_engine.connect() as conn:
            with conn.begin():
                conn.execute(command)

        # Reloading model
        print("RELOADING...")
        global codes_sts, names_sts, texto_sts, tokenized_sts, model_st
        (codes_sts, names_sts, texto_sts, tokenized_sts) = load_solicitacoes()
        model_st = BM25L(text_preprocessed)
        print("RELOAD DONE")
    except:
        return Response(status=500)

    return Response(status=201)


"""Recarga das proposições no modelo
"""
@app.route('/reload-proposicoes', methods=["GET"])
def reloadProposicoes():
    try:
        global model, codes, names, ementas, tokenized_corpus
        (codes, names, ementas, tokenized_corpus) = load_corpus()
        model = BM25L(tokenized_corpus)
        msg = f"Total de proposições recarregadas: {len(codes)}"
    except Exception as ex:
        msg = f"Erro: {str(ex)}"
        print(msg)
        return Response(msg, status=500)
    return Response(msg, status=200)


"""Recarga das solicitações de trabaho no modelo
"""
@app.route('/reload-solicitacoes', methods=["GET"])
def reloadSolicitacoes():
    try:
        global codes_sts, names_sts, texto_sts, tokenized_sts, model_st
        (codes_sts, names_sts, texto_sts, tokenized_sts) = load_solicitacoes()
        model_st = BM25L(tokenized_sts)
        msg = f"Total de solicitações recarregadas: {len(codes_sts)}"
    except Exception as ex:
        msg = f"Erro: {str(ex)}"
        print(msg)
        return Response(msg, status=500)
    return Response(msg, status=200) 


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5000)
