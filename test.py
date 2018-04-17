from flask import Blueprint
from flask_restplus import Api

# api = Api(Blueprint)



from flask_restplus import Api

from api.ner.ner import ns

blueprint = Blueprint('api', __name__)

api = Api(blueprint,
    title='My Title',
    version='1.0',
    description='A description',
    # All API metadatas
)

api.add_namespace(ns)

from flask import Flask

app = Flask(__name__)
# api.init_app(app)
app.register_blueprint(blueprint)

app.run(debug=True)