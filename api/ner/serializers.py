from flask_restplus import fields
from api.restplus import api

ner_result = api.model('entity recognize result', {
    'entity': fields.String(description='entity')
})
