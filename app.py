from __init__ import create_app
import time
from jieba_dict import init_user_dict
from log import log

app = create_app()

# following line used in mod_wsgi deployment mode, app.py act as *.wsgi file directly
application = app

if __name__ == '__main__':
    print('will start server in 20 second')
    time.sleep(20)
    app.run(debug=app.config['FLASK_DEBUG'], host='0.0.0.0', port=28080)
