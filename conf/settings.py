FLASK_ENV = 'prod'  # FLASK deployment env dev/prod


class DefaultSetting(object):
    """Default configuration."""
    # Flask-Restplus settings
    RESTPLUS_SWAGGER_UI_DOC_EXPANSION = 'list'
    RESTPLUS_VALIDATE = True
    RESTPLUS_MASK_SWAGGER = False
    RESTPLUS_ERROR_404_HELP = False

    # SQLAlchemy settings
    # SQLALCHEMY_DATABASE_URI = 'mysql://yxt:pwdasdwx@172.17.128.172:3306/skyeye?charset=utf8'
    # SQLALCHEMY_TRACK_MODIFICATIONS = False
    # SQLALCHEMY_COMMIT_ON_TEARDOWN = True

    # Redis settings
    # REDIS_URL = 'redis://:YDVpwdasdwx2910@172.17.128.186/14'

    # Celery settings
    # CELERY_BROKER_URL = 'redis://:YDVpwdasdwx2910@172.17.128.186/14'
    # CELERY_RESULT_BACKEND = 'redis://:YDVpwdasdwx2910@172.17.128.186/14'


class DevSetting(DefaultSetting):
    """Development configuration."""
    FLASK_DEBUG = True  # Do not use debug mode in production
    FLASK_LOGGER = 'debugLogger'
    FLASK_API_URL_PREFIX = ''  # URL Prefix


class ProdSetting(DefaultSetting):
    """Production configuration."""
    FLASK_DEBUG = False  # Do not use debug mode in production
    FLASK_LOGGER = 'errorLogger'  # Logger
    FLASK_API_URL_PREFIX = '/search_entity/v1/'  # URL Prefix

