#!/usr/bin/env python3

import sys, json, io, base64
import flask
import process
import random

app = flask.Flask(__name__)
app.secret_key = '0b7be479b40b073258d4a9a518acc5a9656b9332f39632c3'
PORT = 1321

sessions = {}

def gen_id():
    while True:
        r = random.randint(10000, 99999)
        if r not in sessions:
            return r

def prepare_page(img_data):
    results = process.process_receipt(img_data, _180=False)
    if not results:
        print('trying 180ed')
        results = process.process_receipt(img_data, _180=True)
        if not results:
            return None
        
    print(len(results))
    
    for result in results:
        result['image'] = base64.b64encode(result['image']).decode()
        
    export_keys = ['price', 'count', 'entry', 'confidence', 'image']
    
    #double json because we are posting this in javascript
    obj = json.dumps(json.dumps([{k: result[k] for k in export_keys} for result in results])).encode()
    
    page = open('template.html', 'rb').read()
    page = page.replace(b'"****obj****"', obj)
    
    return page
    
multipart_start = 'data:image/jpeg;base64,'
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if flask.request.method == 'POST':
        filedata = None
        if 'receipt' in flask.request.form:
            data = flask.request.form['receipt']
            if data.startswith(multipart_start):
                filedata = base64.b64decode(data[len(multipart_start):])
        elif 'receipt' in flask.request.files:
            filedata = flask.request.files['receipt'].read()
        
        if not filedata:
            flask.flash('No file part')
            return flask.redirect(flask.request.url)
            
        page = prepare_page(filedata)
        if not page:
            flask.flash('Bad image')
            return flask.redirect(flask.request.url)
        return page
            
    else:
        return open('index.html', 'rb').read()
    
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=PORT)
