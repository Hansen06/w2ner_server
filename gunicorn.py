import os

workers = int(os.environ.get('WORKER','2'))
bind = '0.0.0.0' + os.environ.get('PORT', '33136')
daemon = 'false'
worker_connections = 1000
timeout = 1200
threads = 2

worker_class = 'gevent'