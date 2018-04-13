from flask import Flask
from flask_cors import *

from api import blueprint
from conf.settings import *


if FLASK_ENV == 'prod':
    DefaultSetting = ProdSetting
    globals()['FLASK_DEBUG'] = False
else:
    DefaultSetting = DevSetting
    globals()['FLASK_DEBUG'] = True


def create_app():

    app = Flask(__name__)
    configure_app(app)
    # register_extensions(app)
    register_blueprints(app)
    return app


def configure_app(app):
    app.config['SWAGGER_UI_DOC_EXPANSION'] = DefaultSetting.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    app.config['RESTPLUS_VALIDATE'] = DefaultSetting.RESTPLUS_VALIDATE
    app.config['RESTPLUS_MASK_SWAGGER'] = DefaultSetting.RESTPLUS_MASK_SWAGGER
    app.config['ERROR_404_HELP'] = DefaultSetting.RESTPLUS_ERROR_404_HELP
    app.config['FLASK_DEBUG'] = DefaultSetting.FLASK_DEBUG


def register_extensions(app):
    CORS(app)
    # db.init_app(app)
    # redis_store.init_app(app)


def register_blueprints(app):
    blueprint.url_prefix = DefaultSetting.FLASK_API_URL_PREFIX
    print(blueprint.url_prefix)
    app.register_blueprint(blueprint)
