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

SELECT_CORPUS_FIELDS = "SELECT code, name, txt_ementa FROM corpus;"
SELECT_ST_FIELDS = "SELECT name, text, text_preprocessed FROM solicitacoes;"
SELECT_CORPUS = "SELECT text_preprocessed FROM corpus;"
SELECT_ST = "SELECT text_preprocessed FROM solicitacoes;"
INSERT_DATA_CORPUS = "INSERT INTO corpus (code, name, txt_ementa, text, text_preprocessed) VALUES ('{}','{}','{}','{}','{}');"
SELECT_ROOT_BY_PROPOSICAO = "SELECT cod_proposicao_raiz FROM arvore_proposicoes WHERE cod_proposicao = {}"
SELECT_AVORE_BY_RAIZ = "SELECT * FROM arvore_proposicoes WHERE cod_proposicao_raiz IN {}"

INSERT_DATA_ST = "INSERT INTO solicitacoes (name, text, text_preprocessed) VALUES ('{}','{}','{}');"
INSERT_DATA_ST_teste = "INSERT INTO solicitacoes (name, text, text_preprocessed) VALUES "
teste2 = "('{}','{}','{}')"

session = requests.Session()
session.trust_env = False
## retrocar ulyssesdb de localhost #####
connection = psycopg2.connect(host="localhost", database="admin",user="admin", password="admin", port=5432)
app = Flask(__name__)

def load_corpus(con):
    with con.cursor() as cursor:
        cursor.execute(SELECT_CORPUS_FIELDS)
        try:
            (codes, names, ementas) = (array(x) for x in transpose( cursor.fetchall() ))
            
            cursor.execute(SELECT_CORPUS)
            tokenized_corpus = cursor.fetchall()
            tokenized_corpus = ["[" + entry[0][1:-1] + "]" for entry in tokenized_corpus]
        except:
            (codes, names, ementas, tokenized_corpus) = [],[],[],[]

        tokenized_corpus = [literal_eval(i) for i in tokenized_corpus]

    print("Loaded", len(names), "documents")
    return (codes, names, ementas, tokenized_corpus)

def load_solicitacoes(con):
    with con.cursor() as cursor:
        cursor.execute(SELECT_ST_FIELDS)
        try:
            (names, texts, tokenized_sts) = (array(x) for x in transpose( cursor.fetchall() ))
            
            #cursor.execute(SELECT_ST)
            #tokenized_sts = cursor.fetchall()
            #tokenized_sts = ["[" + entry[0][1:-1] + "]" for entry in tokenized_sts]
        except:
            (names, texts, tokenized_sts) = [],[],[]

        #tokenized_sts = [literal_eval(i) for i in tokenized_sts]

    print("Loaded", len(names), "Solicitações de Trabalho")
    return (names, texts, tokenized_sts)


# Loading data
print("Loading corpus...")
(codes, names, ementas, tokenized_corpus) = load_corpus(connection)
(names_sts, texto_sts, tokenized_sts) = load_solicitacoes(connection)

# Loading model with dataset
try:
    model = BM25L(tokenized_corpus)
except:
    model = None

try:
    model_st = BM25L(tokenized_sts)
except:
    model_st = None

print("===IT'S ALIVE!===")

def getPastFeedback(con):
    try:
        with con.cursor() as cur:
            cur.execute("SELECT query, user_feedback FROM feedback;")
            (queries, feedbacks) = transpose(cur.fetchall())
        feedbacks = [literal_eval(i) for i in feedbacks]
    except:
        queries, feedbacks = [], []

    scores = []
    all_queries = []
    for entry, q in zip(feedbacks, queries):
        scores.append([[i["id"], float(i["score"]), float(i["score_normalized"])] for i in entry if i["class"]!='i'])
        all_queries.append(q)
    return all_queries, scores

def retrieveDocuments(query, n, raw_query, improve_similarity, con):
    indexes = list(range(len(codes)))

    past_queries, scores = getPastFeedback(con)
    slice_indexes, scores, scores_normalized, scores_final = model.get_top_n(query, indexes, n=n, 
            improve_similarity=improve_similarity, raw_query= raw_query, past_queries=past_queries,
            retrieved_docs=scores, names=names)

    selected_codes = codes[slice_indexes]
    selected_ementas = ementas[slice_indexes]
    selected_names = names[slice_indexes]

    return selected_codes, selected_ementas, selected_names, scores, scores_normalized, scores_final

def retrieveSTs(query, n, raw_query, improve_similarity, con):
    indexes = list(range(len(names_sts)))

    past_queries, scores = getPastFeedback(con)
    slice_indexes, scores, scores_normalized, scores_final = model_st.get_top_n(query, indexes, n=n, 
            improve_similarity=improve_similarity, raw_query= raw_query, past_queries=past_queries,
            retrieved_docs=scores, names=names_sts)

    selected_sts = texto_sts[slice_indexes]
    selected_names = names_sts[slice_indexes]

    return selected_names, selected_sts, scores, scores_normalized, scores_final


def getRelationsFromTree(retrieved_doc):
    with connection.cursor() as cursor:
        cursor.execute(SELECT_ROOT_BY_PROPOSICAO.format(retrieved_doc))
        roots = cursor.fetchall()
        
        # Considerando que documento seja uma raiz
        if (len(roots) == 0):
            roots.append((retrieved_doc,))
        
        roots = [str(i[0]) for i in roots]
        cursor.execute(SELECT_AVORE_BY_RAIZ.format("(%s)" % ",".join(roots)))
        results = cursor.fetchall()
        
        results = list(map(lambda x: {"numero_sequencia": x[0],"nivel": x[1],"cod_proposicao": x[2],
                                     "cod_proposicao_referenciada": x[3],"cod_proposicao_raiz": x[4], 
                                     "tipo_referencia": x[5]}, results))
        return results


@app.route('/', methods=["POST"])
def lookForSimilar(use_relations_tree = False):
    if (model == None):
        print("caimos aqui - test")
        return Response(status=500)

    args = request.json
    print("model aqui - test" + str(model))
    try:
        query = args["text"]
    except:
        return ""
    try:
        k = args["num_proposicoes"]
    except:
        k = 20
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

    k = min(k, len(codes), len(names_sts))

    if (query_expansion):
        resp = session.post("http://expand-query:5003", json={"query": query})
        if (resp.status_code == 200):
            query = json.loads(resp.content)["query"]
    preprocessed_query = preprocess(query)

    # Recuperando das solicitações de trabalho
    selected_names_sts, selected_sts, scores_sts, scores_sts_normalized, scores_final = retrieveSTs(preprocessed_query, k,
                                                                                    improve_similarity=improve_similarity, raw_query=query, con=connection)
    resp_results_sts = list()
    for i  in range(k):
        resp_results_sts.append({"name": selected_names_sts[i], "texto": selected_sts[i].strip(), 
                    "score": scores_sts[i], "score_normalized": scores_sts_normalized[i], "score_final": scores_final[i], "tipo": "ST"})
    

    # Recuperando do corpus das proposições
    selected_codes, selected_ementas, selected_names, scores, scores_normalized, scores_final = retrieveDocuments(preprocessed_query, k, 
                                                                                    improve_similarity=improve_similarity, raw_query=query, con=connection)
    resp_results = list()
    if(use_relations_tree):
        for i  in range(k):
            # Propostas relacionadas pela árvore de proposições
            relations_tree = getRelationsFromTree(selected_codes[i])
            resp_results.append({"id": selected_codes[i], "name": selected_names[i],  
                        "texto": selected_ementas[i].strip(), "score": scores[i], "score_normalized": scores_normalized[i], 
                        "tipo": "PR", "arvore": relations_tree})
    else:
        for i  in range(k):
            resp_results.append({"id": selected_codes[i], "name": selected_names[i],  
                        "texto": selected_ementas[i].strip(), "score": scores[i], "score_normalized": scores_normalized[i], 
                        "tipo": "PR"})        
    
    response = {"proposicoes": resp_results, "solicitacoes": resp_results_sts}
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
        with connection.cursor() as cursor:
            for d in data_insert:
                cursor.execute(INSERT_DATA_CORPUS.format(*d))
            connection.commit()

        # Reloading model
        print("RELOADING...")
        (codes, names, ementas, tokenized_corpus) = load_corpus(connection)
        model = BM25L(tokenized_corpus)
        print("RELOAD DONE")
    except:
        return Response(status=500)

    return Response(status=201)

@app.route('/insert-forced-sts', methods=["POST"]) #inserir planilha sts sem a primeira vigula, antes de name
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
            if(flag):
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
        with connection.cursor() as cursor:
            cursor.execute(command)
            connection.commit()

        # Reloading model
        print("RELOADING...")
        (names, text, text_preprocessed) = load_solicitacoes(connection)
        model_st = BM25L(text_preprocessed)
        print("RELOAD DONE")
    except:
        return Response(status=500)

    return Response(status=201)



if __name__=="__main__":
    ### mudar use_reload para False ###
    app.run(host="0.0.0.0", debug=True, use_reloader=True, port=5000)
