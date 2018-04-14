from flask import request
from flask_restplus import Resource
from api.restplus import api
# from api.ner.business import *
from api.ner.serializers import ner_result

ns = api.namespace('ner', description='recognize search entity')


@ns.route('/<id>')
class Cat(Resource):
    @ns.marshal_with(ner_result)
    def get(self, id):
        '''Fetch a cat given its identifier'''
        return {"entity":'good'}

# @ns.route('/<string:query_content>')
# class SearchEntityRecognizer(Resource):
#
#     @api.marshal_with(ner_result)
#     def get(self, query_content=None):
#         """
#         查询搜索实体
#         """
#         print('vv')
#         return {"entity": "string"}, 200
#
#     # @ns.doc(params={'content': u'查询的字符串'})
