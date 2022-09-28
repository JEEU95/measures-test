from distutils.debug import DEBUG
from operator import truediv
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import pickle
import base64
import io
import os

# create the Flask app
app = Flask(__name__)
PORT = 8000
DEBUG = True
imgFront = None
imgSide = None



@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*" # <- You can change "*" for a domain for example "http://localhost"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization"
    return response

@app.errorhandler(404)
def not_found(error):
    return 'Not found 404'


@app.route('/calculate-measure', methods=['POST'])
@cross_origin()
def calculate_measure():
    request_img = request.get_json()

    if 'view' in request_img:
        if request_img['view'] == 'front'or request_img['view'] == 'side':
            path= 'person_'+ request_img['view']+'.txt'
            file = open(path).read()
            return file
    

    return 'calcular medidas'

@app.route('/define_images', methods=['POST'])
def define_image():
    error=[]
    request_img = request.get_json()
    for req in request_img:
        if req not in ['view','image','base']:
            error.append(req)
    
    if 'view' in request_img and 'image' in request_img and 'base' in request_img:
        name = request_img['image']
        view = request_img['view']
        base = request_img['base'][22:]
        file_name = 'person_'+ view + '.png'
        
        if view == 'front' or view == 'side':
            print(base)

            image = base64.b64decode(base)    
           
            img = Image.open(io.BytesIO(image))
            img.save(file_name)

            return '{"message":"OK"}'
        else:
            return 'Error al ingresar la vista de la imagen'
    return '''Errores({}): Los parametros {} NO existen'''.format(len(error),error)
   
        
if __name__ == '__main__':
    app.run(debug=DEBUG, port=PORT)
    CORS(app, resources={r"*": {"origins": "*"}},support_credentials=True)