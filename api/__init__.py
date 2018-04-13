from flask import Blueprint
from api.restplus import api

from api.ner.ner import ns as ner_namespace

blueprint = Blueprint('api', __name__)
api.init_app(blueprint)
api.add_namespace(ner_namespace)

