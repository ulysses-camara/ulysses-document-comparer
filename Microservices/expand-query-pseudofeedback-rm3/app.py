from flask import Flask, request, jsonify, Response
import requests

from pyserini.search import LuceneSearcher

MAX_EXPANSION = 10
INDEX_PATH = './data_index' #Usar caminho absoluto se necessário


app = Flask(__name__)
searcher = LuceneSearcher(INDEX_PATH)
searcher.set_rm3(10, 10, 0.5)

def rm3_search(query, max_expansion):
    hits = searcher.search(query)
    num_results = min(max_expansion, len(hits))

    if num_results > 0:
        feedback_term = searcher.get_feedback_terms(query)

        sorted_terms = sorted(feedback_term.items(), key=lambda x: x[1], reverse=True)
        result = [{'term':term, 'relevance':relevance} for term, relevance in sorted_terms]
        return result

@app.route('/get_hits', methods=["POST"])
def rm3_get_hits():
    
    args = request.json
    try:
        query = args['query']
    except:
        return Response("Bad Request", status=400)
    try:
        number_of_hits = args["number_of_hits"]
    except:
        number_of_hits = MAX_EXPANSION

    response = {'data': rm3_search(query, number_of_hits)}

    return jsonify(response)

@app.route('/', methods=["POST"])
def rm3_query_expansion():
    args = request.json
    try:
        query = args['query']
    except:
        return Response("Bad Request", status=400)
    try:
        number_of_hits = args["number_of_hits"]
    except:
        number_of_hits = MAX_EXPANSION

    search_results = rm3_search(query, number_of_hits)
    terms = [result['term'] for result in search_results]

    query_expansion = query + ' ' + ' '.join(terms)
    terms_and_relevances = [[result['term'], result['relevance']] for result in search_results]

    response = {'query': query_expansion, 'terms': terms_and_relevances}

    return jsonify(response)

if __name__=="__main__":    
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5005)
