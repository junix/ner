from flask_restplus import Resource

from api.ner.serializers import ner_result
from api.restplus import api

ns = api.namespace('ner', description='recognize search entity')


@ns.route('/<string:query_content>')
@ns.doc(params={'query_content': u'查询的字符串'})
class SearchEntityRecognizer(Resource):

    @api.marshal_with(ner_result)
    def get(self, query_content=None):
        """
        查询搜索实体
        """
        print('vv')
        return {"entity": "string"}
