import re
import traceback
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from CRF import CRF
from BERT import BERT

MODEL_PATH = "pl_st_c_model"

app = Flask(__name__)

api = Api(app, version="1.0", title="API - LookForReferenced", description="Esta documentação refere-se ao microsserviço de detecção e classificação de entidades nomeadas.", doc='/docs')
ns = api.namespace('/', description='Operações do LookForReferenced')
app.config['RESTX_MASK_HEADER'] = False
app.config['RESTX_MASK_SWAGGER'] = False


ner_model = BERT()
try:
    ner_model.set_model_file(MODEL_PATH)
except Exception as ex:
    traceback.print_exc()

print("Modelo carregado com sucesso")


model_request = ns.model('Requisição', {
    'text': fields.String(description="", required=True)
})
model_response = ns.model('Entidades', {
    'entidade': fields.String(description='A entidade detectada.'),
    'tipo': fields.String(description='''Tipo da entidade, pode ser: 
                            FUNDlei -> 'Legal norm';
                            FUNDapelido -> 'Legal norm nickname';
                            FUNDprojetodelei -> 'Bill';
                            FUNDsolicitacaotrabalho -> 'Legislative consultation';
                            LOCALconcreto -> 'Concrete place';
                            LOCALvirtual -> 'Virtual place';
                            ORGpartido -> 'Political party';
                            ORGgovernamental -> 'Gorvernamental organization';
                            ORGnãogovernamental -> 'Non-governamental organization';
                            PESSOAindiviual -> 'Individual';
                            PESSOAgrupoind -> 'Group of individuals';
                            PESSOAcargo -> 'Occupation';
                            PESSOAgrupocargo -> 'Group of occupations';
                            PRODUTOsistema -> 'System product';
                            PRODUTOprograma -> 'Program product';
                            PRODUTOoutros -> 'Others products'.
                            '''),
    'loss': fields.Float(description='Perda associada à entidade no modelo.')
    }, description='Lista de entidades nomeadas encontradas')
@ns.route('/')
class lookForReferenced(Resource):
    @ns.expect(model_request)
    @ns.marshal_list_with(model_response)
    @ns.doc('root', description='Identifica e classifica entidades nomeadas.')
    def post(self):
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
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5003)
