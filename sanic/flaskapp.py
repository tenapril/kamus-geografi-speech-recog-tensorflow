import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, render_template, request, send_file

sio = socketio.Server()
app = Flask(__name__)

client_id = ""

@app.route('/')
def index():
    """Serve the client-side application."""
    return send_file('./static/client.html')

@sio.on('connect')
def connect(sid, environ):
    client_id = sid
    print("connect ", sid)

# @sio.on('chat message')
# def message(sid, data):
#     print("message ", data)
#     sio.emit('reply', room=sid)

@app.route('/stream', methods=['POST'])
def login():
    print(request.get_json())
    sio.emit('datastream', request.get_json())
    return 'OK'

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
