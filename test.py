from flask import Flask

from api import blueprint

app = Flask(__name__)
# api.init_app(app)
app.register_blueprint(blueprint)

app.run(debug=True)
