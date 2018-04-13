import traceback

from flask_restplus import Api
from sqlalchemy.orm.exc import NoResultFound
from log import log

api = Api(version='1.0', title='Search Entity Recognize API',
          description='Search Entity Recognize API')


@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)
    if not globals()['FLASK_DEBUG']:
        return {'message': message}, 500


@api.errorhandler(NoResultFound)
def database_not_found_error_handler(e):
    log.warning(traceback.format_exc())
    return {'message': 'A database result was required but none was found.'}, 404
