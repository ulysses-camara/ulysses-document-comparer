import os
import traceback
import json

from flask import Flask, request, Response, jsonify
from sqlalchemy import create_engine

def table_name(name):
    return "".join([c if c.isalnum() else "_" for c in name])

db_connection = os.getenv('ULYSSES_DB_CONNECTION', default='postgresql+psycopg2://admin:admin@ulyssesdb/admin')
tb_corpus = table_name(os.getenv('TB_CORPUS', default='corpus'))
tb_solicitacoes = table_name(os.getenv('TB_SOLICITACOES', default='solicitacoes'))
tb_feedback = table_name(os.getenv('TB_FEEDBACK', default='feedback'))
db_engine = create_engine(db_connection)

VALID_CLASSES = ["r", "i", "vr", "d", "pr"] # relevante; irrelevante; muito relevante; duvida; pouco relevante

INSERT_ENTRY = "INSERT INTO " + tb_feedback + " (query, user_feedback, extra_results, date_created, user_id) VALUES ({}, {}, {}, CURRENT_TIMESTAMP, {});"
SELECT_ENTRY = "SELECT id, query, user_id, date_created, user_feedback, extra_results FROM " + tb_feedback + " {}"

app = Flask(__name__)


def isResultValid(result):
    try:
        score = float(result["score"])
        score_normalized = float(result["score_normalized"])
        score_final = float(result["score_final"])
        classification = result["class"]
        code = result["id"]
        tipo = result["tipo"]

        if (classification not in VALID_CLASSES):
            return False
    except Exception:
        traceback.print_exc()
        return False
    return True


def isValid(entry):
    try:
        # Parâmetros obrigatórios
        query, results, extra_results = entry["query"], entry["results"], entry["extra_results"]
        if (type(results) != list):
            results = json.loads(results)

        for result in results:
            if (not isResultValid(result)):
                return False
    except Exception:
        traceback.print_exc()
        return False
    return True


@app.route('/', methods=["POST"])
def registerScores():
    data = request.json
    if (isValid(data)):
        try:
            query = data["query"]
            results = data["results"]
            extra_results = data["extra_results"]
            user_id = data.get("user_id")
            json_results = json.dumps(results)
            json_extra_results = json.dumps(extra_results) if extra_results else None
            with db_engine.connect() as conn:
                with conn.begin():
                    conn.execute(
                        INSERT_ENTRY.format(sql_quote(query), sql_quote(json_results), sql_quote(json_extra_results),
                                            sql_quote(user_id)))
            return Response(status=201)
        except Exception:
            traceback.print_exc()
    return Response(status=500)


@app.route('/feedbacks', methods=["GET"])
def feedbacks():
    user_id = request.args.get("user_id")
    sql_criteria = 'WHERE LOWER(user_id) = ' + sql_quote(user_id.lower()) if user_id else ''
    try:
        result = []
        with db_engine.connect() as conn:
            with conn.connection.cursor() as cursor:
                cursor.execute(SELECT_ENTRY.format(sql_criteria))
                for id, query, user_id, date_created, user_feedback, extra_results in cursor:
                    result.append({'id': id, 'query': query, 'user_id': user_id, 'date_created': date_created,
                                   'user_feedback': user_feedback, 'extra_results': extra_results})
        return jsonify(result)
    except Exception:
        traceback.print_exc()
    return Response(status=500)


def sql_quote(text):
    return "'" + text.replace("'", "''") + "'" if text else 'NULL'


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001)