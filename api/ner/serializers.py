from flask_restplus import fields
from api.restplus import api

ner_result = api.model('entity recognize result', {
    'entity': fields.String(description='entity')
})

ner_request = api.model('entity recognize request', {
    'query': fields.String(description='query')
})
