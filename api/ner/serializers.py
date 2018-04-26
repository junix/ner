from flask_restplus import fields
from api.restplus import api

ner_result = api.model('entity recognize result', {
    'entity': fields.String(description='recognized entity'),
    'category': fields.String(description='category of entity')
})

ner_request = api.model('entity recognize request', {
    'query': fields.String(description='query content')
})
