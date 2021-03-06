import os
import json
import re
import datetime
from flask import Flask, request, jsonify, Response
import requests
import psycopg2

import base64
from cryptography.fernet import Fernet

FIND_BY_NAME_CORPUS = 'SELECT txt_ementa FROM corpus WHERE name IN %s'
FIND_BY_NAME_ST = 'SELECT text FROM solicitacoes WHERE name IN %s'

PL_REGEX = "[0-9]+"
LABELS = ["ADD","ANEXO","APJ","ATC","AV","CN","EMS","INC","MPV","MSC","PL","PEC","PLP",
            "PLV","PDC","PRC","PRN","PFC","REP","REQ","RIC","RCP","SIT","ST"]

url_look_for_referenced = os.getenv('URL_LOOK_FOR_REFERENCED', default='http://look-for-referenced:5002')
crypt_key = os.getenv('CRYPT_KEY_SOLIC_TRAB', default='')
net = Fernet(crypt_key.encode()) if crypt_key else None
db_connection = os.getenv('ULYSSES_DB_CONNECTION', default='host=ulyssesdb dbname=admin user=admin password=admin port=5432')
app = Flask(__name__)

print("Microsserviço iniciado com sucesso")

# Retrieves the text from the corpus and the STs based on the referenced name
def searchByName(name_parts, cursor):

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

        cursor.execute(FIND_BY_NAME_CORPUS, (labeled_codes,))
        results = [text[0] for text in cursor.fetchall()]
        if query_expansion == "":
            query_expansion = ' '.join(results)
        else:
            query_expansion += ' ' + ' '.join(results)

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
    except:
        return Response(status=500)

    resp = requests.post(url_look_for_referenced, json={"text": query})
    data = json.loads(resp.content)

    conn = psycopg2.connect(db_connection)
    try:
        with conn:
            with conn.cursor() as cursor:
                for entity in data["entities"]:
                    string, score = entity[0], entity[1]
                    name_parts = re.findall(PL_REGEX, string)
                    expansion = searchByName(name_parts, cursor)
                    query += " " + expansion
                    query = query.strip()
    finally:
        conn.close()

    resp = {'query': query, 'entities': data["entities"]}
    return jsonify(resp)


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5003)