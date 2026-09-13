#创建该文件的主要目的是为了解决Flask自带的app.run()是单线程的问题，如果外部公司同时发来两个请求，它会直接卡死甚至崩溃。需要用 Gunicorn 这个工业级 WSGI 服务器来托管它。
# gunicorn_config.py
bind = "0.0.0.0:8099"
workers = 1
threads = 4
timeout = 3600