from flask import request
from flask_restplus import Resource
from api.restplus import api
# from api.ner.business import *
from api.ner.serializers import ner_result

ns = api.namespace('ner', description='recognize search entity')


@ns.route('/<string:query_content>')
class SearchEntityRecognizer(Resource):

    def post(self):
        """
        Update list of tags.
        """
        print('vv')
        return {"entity": "string"}, 200




    # @api.marshal_with(ner_result)
    # @ns.doc(params={'content': u'查询的字符串'})
