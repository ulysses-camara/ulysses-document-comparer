import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet

import json
import re
from flask import Flask, request, jsonify, Response
import requests

nlp = spacy.load("pt_core_news_sm")
app = Flask(__name__)
print("Microsserviço iniciado com sucesso")

def get_word_synonyms(word):
    synonyms = []
    for synset in wordnet.synsets(word, lang='por'):
        for lemma in synset.lemmas(lang='por'):
            synonyms.append(lemma.name())
    return synonyms

@app.route('/get_synonyms', methods=["GET"])
def get_word_list_synonyms():
    
    args = request.args
    try:
        query = args["words"]
    except:
        return Response("Bad Request", status=400)
    
    words = query.split(",")
    print(type(words))
    data = []
    for word in words:
        data.append({"key": word, "synonyms": get_word_synonyms(word)})
    return jsonify({"data": data})

@app.route('/', methods=["POST"])
def get_noun_synonyms():
    synonyms_list = []

    args = request.json
    try:
        query = args["query"]
    except:
        return Response("Bad Request", status=400)

    sentence = query
    doc = nlp(sentence)

    entities_synonyms = []
    for token in doc:
        # if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:  # Filtrar substantivos, verbos, adjetivos e advérbios

        if token.pos_ == 'NOUN':  # Filtrar apenas os substantivos
            token_synonyms = get_word_synonyms(token.text.lower())
            cleaned_synonyms = [synonym.lower() for synonym in token_synonyms if '_' not in synonym]
            unique_synonyms = list(set(cleaned_synonyms))
            synonyms_list.extend( unique_synonyms[:3])
            entities_synonyms.extend([[synonym.lower(), token.text.lower()] for synonym in unique_synonyms[:3]])

    synonyms_list = list(set([value.lower() for value in synonyms_list]))
    result =  " ".join(synonyms_list)
    final_values = sentence + " " + result
    return jsonify({"query": final_values, 'entities_synonyms': entities_synonyms})

if __name__=="__main__":    
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5004)