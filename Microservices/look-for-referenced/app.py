import re
import traceback
from flask import Flask, request, jsonify
from CRF import CRF

app = Flask(__name__)

ner_model = CRF()
try:
    ner_model.set_model_file("model.crf.tagger.app")
except Exception as ex:
    traceback.print_exc()

print("Modelo carregado com sucesso")

@app.route('/', methods=["POST"])
def lookForReferenced():
    args = request.json
    query = args["text"]
    tokenized_query = re.findall(r"[\w']+|[.,!?;]", query)
    try:
        named_entities = ner_model.return_docs(tokenized_query)
        response = {"entities": named_entities}
        return jsonify(response)
    except Exception as ex:
        traceback.print_exc()

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5002)