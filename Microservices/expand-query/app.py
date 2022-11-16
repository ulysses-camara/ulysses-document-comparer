import os
import json
import re
import datetime
import base64
import requests

from flask import Flask, request, jsonify, Response
from sqlalchemy import create_engine
from cryptography.fernet import Fernet

def table_name(name):
    return "".join([c if c.isalnum() else "_" for c in name])

db_connection = os.getenv('ULYSSES_DB_CONNECTION', default='postgresql+psycopg2://admin:admin@ulyssesdb/admin')
tb_corpus = table_name(os.getenv('TB_CORPUS', default='corpus'))
tb_solicitacoes = table_name(os.getenv('TB_SOLICITACOES', default='solicitacoes'))
tb_feedback = table_name(os.getenv('TB_FEEDBACK', default='feedback'))
db_engine = create_engine(db_connection)


#Possivelmente devem ser corrigidos devido ao refatoramento do bd
FIND_BY_NAME_CORPUS = 'SELECT txt_ementa FROM ' + tb_corpus + ' WHERE name IN %s'
FIND_BY_NAME_ST = 'SELECT text FROM ' + tb_solicitacoes + ' WHERE name IN %s'

PL_REGEX = "[0-9]+" #Busca por números, pode ser usado nas leis e nas STs também, mas não para os FUNDapelidos
LABELS = ["ADD","ANEXO","APJ","ATC","AV","CN","EMS","INC","MPV","MSC","PL","PEC","PLP",
            "PLV","PDC","PRC","PRN","PFC","REP","REQ","RIC","RCP","SIT","ST"]

url_look_for_referenced = os.getenv('URL_LOOK_FOR_REFERENCED', default='http://look-for-referenced:5002')
crypt_key = os.getenv('CRYPT_KEY_SOLIC_TRAB', default='')
net = Fernet(crypt_key.encode()) if crypt_key else None

app = Flask(__name__)

print("Microsserviço iniciado com sucesso")

# Retrieves the text from the corpus and the STs based on the referenced name
def searchByName(name_parts, label, cursor):

    query_expansion = ""
    if (len(name_parts)==2):
        code = name_parts[0]
        code_year = name_parts[1]
        if (len(code_year) == 2):
            if (int(code_year) <= (datetime.now().year % 100)):  # Cheking for 2-digit year (ex 2019 -> 19, 1998 -> 98)
                code_year = "20"+code_year
            else:
                code_year = "19"+code_year
        code = code + "/" + code_year

        labeled_codes = tuple(map(lambda x: x + " " + code, LABELS))

        if label == 'FUNDprojetodelei':
            cursor.execute(FIND_BY_NAME_CORPUS, (labeled_codes,))
            results = [text[0] for text in cursor.fetchall()]
            if query_expansion == "":
                query_expansion = ' '.join(results)
            else:
                query_expansion += ' ' + ' '.join(results)

        elif label == 'FUNDsolicitacaotrabalho':
            cursor.execute(FIND_BY_NAME_ST, (labeled_codes,))
            results = [net.decrypt(base64.b64decode(text[0])).decode() if net else text[0] for text in
                    cursor.fetchall()]
            if query_expansion == "":
                query_expansion = ' '.join(results)
            else:
                query_expansion += ' ' + ' '.join(results)

    return query_expansion
            

@app.route('/', methods=["POST"])
def queryExpansion():

    args = request.json
    try:
        query = args["query"]
        resp = requests.post(url_look_for_referenced, json={"text": query})
        data = json.loads(resp.content)
    except:
        return Response(status=500)

    #labels de interesse: FUNDlei(?), FUNDapelido(?), FUNDprojetodelei e FUNDsolicitacaotrabalho
    with db_engine.connect() as conn:
        with conn.connection.cursor() as cursor:
            for entity in data["entities"]:
                token, label, score = entity[0], entity[1], entity[2]
                if label == 'FUNDprojetodelei' or label == 'FUNDsolicitacaotrabalho':
                    name_parts = re.findall(PL_REGEX, token)
                    expansion = searchByName(name_parts, label, cursor)
                    query += " " + expansion
                    query = query.strip()

    resp = {'query': query, 'entities': data["entities"]}
    return jsonify(resp)


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5003)