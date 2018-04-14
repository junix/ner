from flask_restplus import Api

from log import log

api = Api(version='1.0', title='Search Entity Recognize API', description='Search Entity Recognize API')


@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)
    if not globals()['FLASK_DEBUG']:
        return {'message': message}, 500



