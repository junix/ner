from flask_restplus import Resource
from flask import request

from api.ner.business import get_entity
from api.ner.serializers import ner_result, ner_request
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
        if query_content is None:
            return {"entity": ""}

        return {"entity": get_entity(query_content)}


@ns.route('/')
class BatchSearchEntityRecognizer(Resource):
    @api.expect([ner_request])
    @api.marshal_list_with(ner_result)
    def post(self):
        """
        查询搜索实体
        """
        queries = [item['query'] for item in request.json]
        reps = [{'entity': get_entity(q)} for q in queries]
        return reps, 201
