import re
import traceback
from flask import Flask, request, jsonify
from CRF import CRF
from BERT import BERT

app = Flask(__name__)

ner_model = BERT()
try:
    ner_model.set_model_file("pl_st_c_model")
except Exception as ex:
    traceback.print_exc()

print("Modelo carregado com sucesso")

@app.route('/', methods=["POST"])
def lookForReferenced():
    args = request.json
    query = args["text"]
    #tokenized_query = re.findall(r"[\w']+|[.,!?;]", query)
    tokenized_query = query.split()
    try:
        named_entities = ner_model.return_entities(tokenized_query)
        response = {"entities": named_entities}
        return jsonify(response)
    except Exception as ex:
        traceback.print_exc()

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5002)