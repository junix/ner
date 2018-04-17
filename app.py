from __init__ import create_app
from api.ner.business import get_entity

app = create_app()

# following line used in mod_wsgi deployment mode, app.py act as *.wsgi file directly
application = app

if __name__ == '__main__':
    # get_entity("s")
    app.run(debug=app.config['FLASK_DEBUG'], host='0.0.0.0', port=8888)
