#!/usr/bin/env python3

import sys, json, io, base64, random, datetime, inspect
import flask
from flask_restful import Api, Resource
import process

app = flask.Flask(__name__)
api = Api(app)
app.secret_key = '0b7be479b40b073258d4a9a518acc5a9656b9332f39632c3'
PORT = 1321

sessions = {}
session_exports = {}

def export(f):
    spec = inspect.getfullargspec(f)
    args = {arg: spec.annotations.get(arg, None) for arg in spec.args[1:]} #because of self
    session_exports[f.__name__] = args
    return f

class Session:
    
    def gen_id(self):
        self.c += 1
        return self.c
        

    @classmethod
    def gen_session_id(cls):
        while True:
            r = random.randint(10000, 99999)
            if r not in sessions:
                return r
            
    def __init__(self, obj):
        global sessions
        
        self.c = 0
        self.session_id = self.gen_session_id()
        self.t = datetime.datetime.now()
        
        consumers = 8
                
        self.obj = obj
        self.consumers = {}
        self.items = {}
        self.instances = {}
        self.misc = {}
        
        for i in range(consumers):
            consumer_id = self.gen_id()
            self.consumers[consumer_id] = dict(consumer_id=consumer_id, name='')
            
        for i, e in enumerate(obj):
            self.new_item_internal(e['price'], e['entry'], e['confidence'], e['image'], e['count'], i)
            
        sessions[self.session_id] = self
    
    def new_item_internal(self, price, entry, confidence, image, count, obj_index):
        item_id = self.gen_id()
        self.items[item_id] = dict(item_id=item_id, price=price, entry=entry, confidence=confidence, image=image, obj_index=obj_index)
        for _ in range(count):
            self.new_instance(item_id)
            
    @export
    def new_item(self, price: float, entry: str):
        self.new_item_internal(price, entry, -1, b'', 1, -1)
        
    @export
    def new_instance(self, item_id: int):
        instance_id = self.gen_id()
        self.instances[instance_id] = dict(instance_id=instance_id, item_id=item_id, consumer_id=-1)
        
    #no new consumer for now
    
    @export
    def remove_instance(self, instance_id: int):
        if instance_id not in self.instances:
            return
        del self.instances[instance_id]
    
    @export
    def reset_item(self, item_id: int):
        if item_id not in self.items:
            return
        obj_index = self.items[item_id]['obj_index']
        if obj_index == -1:
            #not original, reset is delete
            del self.items[item_id]
        else:
            #original, reset
            item = self.items[item_id]
            orig = self.obj[obj_index]
            item['price'] = orig['price']
            item['entry'] = orig['entry']
            item['confidence'] = orig['confidence']
            item['image'] = orig['image']
            
        #delete all instances
        instance_ids = [k for k, v in self.instances.items() if v['item_id'] == item_id]
        for instance_id in instance_ids:
            del self.instances[instance_id]
            
    @export
    def reset_consumer(self, consumer_id: int):
        #reset all instances
        for k,v in self.instances.items():
            if v['consumer_id'] == consumer_id:
                v['consumer_id'] = -1
                
    @export
    def change_instance(self, instance_id: int, consumer_id: int):
        if instance_id not in self.instances or consumer_id not in self.consumers:
            return
        self.instances[instance_id] = consumer_id
        
    @export
    def change_consumer(self, consumer_id: int, name: str):
        if consumer_id not in self.consumers:
            return
        self.consumers[consumer_id]['name'] = name
           
    @export
    def change_item(self, item_id: int, price: float, entry: str):
        if item_id not in self.items:
            return
        if price != '':
            self.items[item_id]['price'] = price
        if entry != '':
            self.items[item_id]['entry'] = entry
            
    @export
    def set_misc(self, key: str, value):
        self.misc[key] = value
            
    def get(self, no_images=False):
        obj = dict(session_id=self.session_id,
                        consumers=list(self.consumers.values()),
                        items=[{k:v for k, v in item.items() if k != 'image' or not no_images} for item in self.items.values()],
                        instances=list(self.instances.values()),
                        misc=self.misc)
        
        print(f'getting {self.session_id}')
        return obj, 200
                        
    def put(self, method, **kwargs):
        print(f'putting {self.session_id} {method} {kwargs}')
        
        if method not in session_exports:
            return method, 404
            
        if set(session_exports[method].keys()) != set(kwargs.keys()):
            return repr(session_exports[method].keys()), 400
        
        print(session_exports[method])
        try:
            converted_args = {k: (c(kwargs[k]) if c is not None else kwargs[k]) for k, c in session_exports[method].items()}
        except:
            return repr(session_exports[method].keys()), 400
        #try:        
        getattr(self, method)(**converted_args)
        #except Exception as e:
        #    print(e)
        #    return method, 500
        return '', 200
        
class Router(Resource):
    def get(self, session_id, method):
        if session_id not in sessions:
            return '', 404
        return sessions[session_id].get(no_images=method=='no_images')
        
    def put(self, session_id, method):
        if session_id not in sessions:
            return '', 404
        return sessions[session_id].put(method, **{k: v for k, v in flask.request.args.items()})

def analyze_image(img_data):
    results = process.process_receipt(img_data, _180=False)
    if not results:
        print('trying 180ed')
        results = process.process_receipt(img_data, _180=True)
        if not results:
            return None
    
    for result in results:
        result['image'] = base64.b64encode(result['image']).decode()
        
    export_keys = ['price', 'count', 'entry', 'confidence', 'image']
    
    return Session([{k: result[k] for k in export_keys} for result in results]).get()
    
def prepare_page(obj):
    #double json because we are posting this in javascript
    js = json.dumps(json.dumps(obj)).encode()
    
    page = open('template.html', 'rb').read()
    page = page.replace(b'"****obj****"', js)
    
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
            
        page = prepare_page(analyze_image(filedata)[0])
        if not page:
            flask.flash('Bad image')
            return flask.redirect(flask.request.url)
        return page
            
    else:
        return open('index.html', 'rb').read()
    
api.add_resource(Router, '/ajax/<int:session_id>/<method>')

if __name__ == '__main__':
    #from waitress import serve
    #serve(app, host='0.0.0.0', port=PORT)
    app.run(debug=True)
