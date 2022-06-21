import os
import traceback

from flask import Flask, request, Response
import psycopg2
import json

VALID_CLASSES = ["r", "i", "vr", "d", "pr"] # relevante; irrelevante; muito relevante; duvida; pouco relevante
INSERT_ENTRY = "INSERT INTO feedback(query, user_feedback, extra_results, date_created, user_id) VALUES ({}, {}, {}, CURRENT_TIMESTAMP, {});"

app = Flask(__name__)

db_connection = os.getenv('ULYSSES_DB_CONNECTION', default='host=ulyssesdb database=admin user=admin password=admin port=5432')
connection = psycopg2.connect(db_connection)

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
    except Exception as ex:
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
    except Exception as ex:
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
            with connection.cursor() as cursor:
                try:
                    cursor.execute(INSERT_ENTRY.format(sql_quote(query), sql_quote(json_results), sql_quote(json_extra_results), sql_quote(user_id)))
                    connection.commit()
                except Exception as ex:
                    connection.rollback()
                    raise ex
            return Response(status=201)
        except Exception as ex:
            traceback.print_exc()
    return Response(status=500)

def sql_quote(text):
    return "'" + text.replace("'", "''") + "'" if text else 'NULL'

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001)