from flask_restplus import Api

from api.ner.ner import ns
api = Api(
    title='My Title',
    version='1.0',
    description='A description',
    # All API metadatas
)

api.add_namespace(ns)

from flask import Flask

app = Flask(__name__)
api.init_app(app)

app.run(debug=True)