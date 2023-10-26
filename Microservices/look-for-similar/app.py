import os
from ast import literal_eval
import json
import requests
from flask import Flask, request, jsonify, Response
from flask_restx import Api, Resource, fields
from numpy import transpose, array
from bm25 import BM25L, DEFAULT_CUT, DEFAULT_DELTA, DEFAULT_PESO_POUCO_RELEVANTES
from preprocessing import preprocess
import base64
from cryptography.fernet import Fernet
from sqlalchemy import create_engine

from time import time


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
SELECT_FEEDBACK = f"SELECT query, user_feedback FROM {tb_feedback};"

INSERT_DATA_ST = "INSERT INTO " + tb_solicitacoes + " (name, text, text_preprocessed) VALUES ('{}','{}','{}');"
INSERT_DATA_ST_teste = "INSERT INTO " + tb_solicitacoes + " (name, text, text_preprocessed) VALUES "

app = Flask(__name__)

api = Api(app, version="1.0", title="API - LookForSimilar", description="Esta documentação refere-se ao microsserviço de busca de entidades similares.", doc='/docs')
ns = api.namespace('/', description='Operações da LookForSimilar')
app.config['RESTX_MASK_HEADER'] = False
app.config['RESTX_MASK_SWAGGER'] = False

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
print("Carregando proposições da base de dados...")
start_time = time()
#(codes, names, ementas, tokenized_corpus) = load_corpus()
print(f'Tempo de carga da base de dados (proposições): {round(time() - start_time, 2)} s.')

print("Carregando solicitações da base de dados...")
start_time = time()
#(codes_sts, names_sts, texto_sts, tokenized_sts) = load_solicitacoes()
print(f'Tempo de carga da base de dados (solicitações): {round(time() - start_time, 2)} s.')

# Loading model with dataset
print('Carregando BM25L para proposições...')
start_time = time()
#model = BM25L(tokenized_corpus)
print(f'Tempo de carga BM25L (proposições): {round(time() - start_time, 2)} s.')

print('Carregando BM25L para solicitações...')
start_time = time()
#model_st = BM25L(tokenized_sts)
print(f'Tempo de carga BM25L (solicitações): {round(time() - start_time, 2)} s.')

print("Modelos carregados com sucesso!", end='\n\n')


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
        scores.append([[i["id"], float(i["score"]), float(i["score_normalized"]), i["class"]] for i in
                       entry])  # if i["class"]!='i'])
        all_queries.append(q)
    return all_queries, scores


def retrieveDocuments(query, n, raw_query, improve_similarity, past_queries, past_scores,
                      passed_cut=DEFAULT_CUT, passed_delta=DEFAULT_DELTA,
                      peso_pouco_relevantes=DEFAULT_PESO_POUCO_RELEVANTES):
    indexes = list(range(len(codes)))
    slice_indexes, scores, scores_normalized, scores_final = model.get_top_n(
        query, indexes, n=n, improve_similarity=improve_similarity, raw_query=raw_query, past_queries=past_queries,
        retrieved_docs=past_scores, names=names, cut=passed_cut, delta=passed_delta,
        peso_pouco_relevantes=peso_pouco_relevantes)
    selected_codes = codes[slice_indexes]
    selected_ementas = ementas[slice_indexes]
    selected_names = names[slice_indexes]
    return selected_codes, selected_ementas, selected_names, scores, scores_normalized, scores_final


def retrieveSTs(query, n, raw_query, improve_similarity, past_queries, past_scores,
                passed_cut=DEFAULT_CUT, passed_delta=DEFAULT_DELTA,
                peso_pouco_relevantes=DEFAULT_PESO_POUCO_RELEVANTES):
    indexes = list(range(len(names_sts)))
    slice_indexes, scores, scores_normalized, scores_final = model_st.get_top_n(
        query, indexes, n=n, improve_similarity=improve_similarity, raw_query=raw_query, past_queries=past_queries,
        retrieved_docs=past_scores, names=names_sts, cut=passed_cut, delta=passed_delta,
        peso_pouco_relevantes=peso_pouco_relevantes)
    selected_sts = texto_sts[slice_indexes]
    selected_names = names_sts[slice_indexes]
    selected_codes = codes_sts[slice_indexes]
    return selected_codes, selected_names, selected_sts, scores, scores_normalized, scores_final


request_model = ns.model('Requisição', {
    'num_suposicoes': fields.Integer(description='O número de suposições máximas retornadas.', default=20),
    'num_proposicoes': fields.Integer(description='Numero de proposições máximas retornadas.', default=20),
    'text': fields.String(description='A query, ex., "Gostaria de consultar todos os documentos relacionados a PL XXXX/YYYY".'),
    'expansao': fields.Integer(description='Indica se a query deve ser expandida (pode melhorar os resultados da pesquisa), 0 (não) demais (sim).', default=1),
    'improve_similarity': fields.Integer(description="Indica se o modelo deve usar a 'ImproveSimilarity' (pode melhorar os resultados da pesquisa), 0 (não) demais (sim).", default=1),
    'cut': fields.Float(description="Define um limiar no qual documentos são considerados similares pela 'ImproveSimilarity', deve estar na faixa de -1 (menos similar) e +1 (mais similar).", default=DEFAULT_CUT),
    'delta': fields.Float(description="Define um fator de escala para documentos recuperados pela 'ImproveSimilarity', quanto maior, mais impacto estes documentos tem no resultado final.", default=DEFAULT_DELTA),
})

response_model_prep = ns.model('Proposição', {
    'id': fields.Integer(description='ID sintético da preprosição.'),
    'name': fields.String(description='Nome da preposição.'),
    'texto': fields.String(description='Texto completo da ementa da preposição.'),
    'score': fields.Integer(description='Score inicial do documento, com base na frequência de termos.'),
    'score_normalized': fields.Integer(description='Score normalizado de 0 a 1, onde o documento mais relevante possui um score de 1 e o menos relevante de 0.'),
    'score_final':  fields.Integer(description="Score final atualizado com os resultados da 'ImproveSimilarity', sera igual ao 'score_normalized' se não habilitada."),
    'tipo': fields.String('PR')
})
response_model_st = ns.model('Suposição', {
    'id': fields.Integer(description='ID sintético da suposição.'),
    'name': fields.String(description='Nome da suposição.'),
    'texto': fields.String(description='Texto completo da suposição.'),
    'score': fields.Integer(description='Score inicial do documento, com base na frequência de termos.'),
    'score_normalized': fields.Integer(description='Score normalizado de 0 a 1, onde o documento mais relevante possui um score de 1 e o menos relevante de 0.'),
    'score_final':  fields.Integer(description="Score final atualizado com os resultados da 'ImproveSimilarity', sera igual ao 'score_normalized' se não habilitada."),
    'tipo': fields.String('ST')
})
response_model = ns.model('Resposta', {
    'proposicoes': fields.List(fields.Nested(response_model_prep), description="Uma lista de 'num_proposicoes' proposições encontradas e ordenadas."),
    'solicitacoes': fields.List(fields.Nested(response_model_st), description="Uma lista de 'num_solicitacoes' solicitações encontradas e ordenadas."),
    'actual_query': fields.String(description='A query utilizada.'),
})
@ns.route('/')
class lookForSimilar(Resource):
    @ns.doc('root', description='Busca por proposições e suposições semelhantes a uma dada query.')
    @ns.expect(request_model)
    @ns.marshal_with(response_model)
    def post(self):
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

        try:
            passed_cut = float(args["cut"])
        except:
            passed_cut = DEFAULT_CUT
        try:
            passed_delta = float(args["delta"])
        except:
            passed_delta = DEFAULT_DELTA

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
        selected_codes_sts, selected_names_sts, selected_sts, scores_sts, scores_sts_normalized, scores_sts_final = retrieveSTs(
            preprocessed_query, k_st,
            improve_similarity=improve_similarity, raw_query=query,
            past_queries=past_queries, past_scores=past_scores,
            passed_cut=passed_cut, passed_delta=passed_delta)
        resp_results_sts = list()
        for i in range(k_st):
            resp_results_sts.append(
                {"id": int(selected_codes_sts[i]), "name": selected_names_sts[i], "texto": selected_sts[i].strip(),
                "score": scores_sts[i], "score_normalized": scores_sts_normalized[i],
                "score_final": scores_sts_final[i], "tipo": "ST"})

        # Recuperando do corpus das proposições
        selected_codes, selected_ementas, selected_names, scores, scores_normalized, scores_final = retrieveDocuments(
            preprocessed_query, k_prop,
            improve_similarity=improve_similarity, raw_query=query,
            past_queries=past_queries, past_scores=past_scores,
            passed_cut=passed_cut, passed_delta=passed_delta)
        resp_results = list()
        for i in range(k_prop):
            resp_results.append({"id": int(selected_codes[i]), "name": selected_names[i],
                                "texto": selected_ementas[i].strip(), "score": scores[i],
                                "score_normalized": scores_normalized[i],
                                "score_final": scores_final[i], "tipo": "PR"})
        response = {"proposicoes": resp_results, "solicitacoes": resp_results_sts, "actual_query": query}
        return jsonify(response)


"""Recarga das proposições no modelo
"""


@ns.route('/reload-proposicoes')
class reloadProposicoes(Resource):
    @ns.doc('reload-proposicoes', description='Recarrega o dadaset de proposições.', responses={200: 'OK', 500: 'Erro Interno do Servidor'})
    def post(self):
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


@ns.route('/reload-solicitacoes')
class reloadSolicitacoes(Resource):
    @ns.doc('reload-solicitacoes', description='Recarrega o dadaset de solicitações.', responses={200: 'OK', 500: 'Erro Interno do Servidor'})
    def post(self):
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5000)
