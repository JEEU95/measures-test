from distutils.debug import DEBUG
from email.mime import image
from operator import truediv
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import pickle
import base64
import io
import os

import cv2
import mediapipe as mp
import numpy as np

# create the Flask app
app = Flask(__name__)
PORT = 8000
DEBUG = True

height_cm = 0
height_pxl = 0
measures_cm = {'Shoulders': 0,'Arms': 0,'Chest': 0,'hips': 0,'Legs': 0,'Feet': 0}
measures_pxl = {'Shoulders': 0,'Arms': 0,'Chest': 0,'hips': 0,'Legs': 0,'Feet': 0}
measures_front= {'height': 0,
               'shoulders': 0,
               'hip': 0,
               'chest': 0,
               'arms': 0,
               'forearms': 0,
               'legs': 0,
               'thighs': 0,
               'ears': 0,
               'foots': 0
               }
measures_side= {'height': 0,
               'shoulders': 0,
               'hip': 0,
               'chest': 0,
               'arms': 0,
               'forearms': 0,
               'legs': 0,
               'thighs': 0,
               'ears': 0,
               'foots': 0
               }

imgFront = None
imgSide = None
respImage = {'view': 'string','image': 'string','base': 'string'}

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
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
    global height_cm
    request_measure = request.get_json()


    if 'name' in request_measure and 'measure_cm' in request_measure:
        if request_measure['name'] == 'height':
            height_cm = request_measure['measure_cm']
            #print(height_cm)
            measures()
        resp = measures_cm
        """ print(respImage)
        print(resp) """
        return [respImage,resp]
    return '{"message":"Error en los parametros"}'

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

            image = base64.b64decode(base)    
           
            img = Image.open(io.BytesIO(image))
            img.save(file_name)

            return '{"message":"OK"}'
        else:
            return 'Error al ingresar la vista de la imagen'
    return '''Errores({}): Los parametros {} NO existen'''.format(len(error),error)
   
def dibujar_contornos(image,n):
    bordeCanny = cv2.Canny(image,100,100)
    ctns, _ = cv2.findContours(bordeCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, ctns, -1, (0, 0, 255), 2)

    bordes = cv2.bitwise_not(bordeCanny)
    archivo='assets/bordes'+str(n)+'.jpg'

    cv2.imwrite(archivo,bordes)
    
    #cv2.imshow('canny'+str(n),image)
    return bordes


def def_height(image):
    global height_pxl
    #print(height_pxl)
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    rows, cols = image.shape

    for x in range(rows):
        for y in range(cols):
            if image[x,y] == 0:
                if x1 == 0 and y1 == 0:
                    x1 = y
                    y1 = x
                else:
                    x2 = y
                    y2 = x
    height = y2 - y1
    height_pxl = height
    color = (0, 255, 0)
    img = cv2.imread("assets/bordes1.jpg")

    img = cv2.line(img, (x1, y1), (x1, y2), color, 3)
    #cv2.imshow('height', img)

def calcular_distancia(p1, p2):

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    res = (((x2 - x1) ** 2)+ ((y2 - y1) ** 2))**(1 / 2)

    return round(abs(res))

'''mascara para obtener solo la persona'''
def segmentation(img):
    results_mask = []
    BG_COLOR = (192, 192, 192)  # gray
    MASK_COLOR = (255, 255, 255)  # white
    with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=0) as selfie_segmentation:
        image = img
        height, width, _ = image.shape
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        condition = np.stack((results.segmentation_mask,) , axis=-1) > 0.1
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        output_image = np.where(condition, image, bg_image)

        output_image2 = np.where(condition, fg_image, bg_image)
        '''cv2.imshow('SNmask1', output_image) 
        cv2.imshow('SNmask2', output_image2)'''
        mask = cv2.medianBlur(output_image2, 13)
        output_image2 = mask

        x,y, _ = mask.shape
 
        for i in range(x):
            for j in range(y):
                if np.array_equal(mask[i,j], np.array([192, 192, 192])) == True:
                    output_image[i, j] = np.array([192, 192, 192]) 

        
        mask = cv2.medianBlur(output_image, 3)
        output_image = mask

        '''cv2.imshow('mask1', output_image) 
        cv2.imshow('mask2', output_image2) '''

        results_mask.append([output_image,dibujar_contornos(output_image,0)])

        results_mask.append([output_image2, dibujar_contornos(output_image2,1)])

        '''cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        return results_mask

def crear_dict(img, landmarks):
    dict_landmarks = {}
    image = img
    height, width, _ = image.shape

    for i in range(0, 33):
        var = ''
        if i == 0:
            var = 'nose'
        elif i >= 1 and i <= 3:
            var = 'left_eye'
            if i == 1:
                var = var + '_inner'
            if i == 3:
                var = var + '_outer'
        elif i > 3 and i <= 6:
            var = 'right_eye'
            if i == 4:
                var = var + '_inner'
            if i == 6:
                var = var + '_outer'
        if i >= 7:
            if i % 2 == 0:
                var = 'right_'
            else:
                var = 'left_'
            if i >= 7 and i <= 8:
                var = var + 'ear'
            if i >= 9 and i <= 10:
                var = var + 'mouth'
            if i >= 11 and i <= 12:
                var = var + 'shoulder'
            if i >= 13 and i <= 14:
                var = var + 'elbow'
            if i >= 15 and i <= 16:
                var = var + 'wrist'
            if i >= 17 and i <= 18:
                var = var + 'pinky'
            if i >= 19 and i <= 20:
                var = var + 'index'
            if i >= 21 and i <= 22:
                var = var + 'thumb'
            if i >= 23 and i <= 24:
                var = var + 'hip'
            if i >= 25 and i <= 26:
                var = var + 'knee'
            if i >= 27 and i <= 28:
                var = var + 'ankle'
            if i >= 29 and i <= 30:
                var = var + 'heel'
            if i >= 31 and i <= 32:
                var = var + 'foot_index'
        dict_landmarks[var] = [int(landmarks.landmark[i].x * width), int(landmarks.landmark[i].y * height)]
    return dict_landmarks


def postura(image,sombra):
    medidas = {'height': height_pxl,
               'shoulders': 0,
               'hip': 0,
               'chest': 0,
               'arms': 0,
               'forearms': 0,
               'legs': 0,
               'thighs': 0,
               'ears': 0,
               'foots': 0
               }
    medidas_cm = {'height': height_cm,
               'shoulders': 0,
               'hip': 0,
               'chest': 0,
               'arms': 0,
               'forearms': 0,
               'legs': 0,
               'thighs': 0,
               'ears': 0,
               'foots': 0
               }
    with mp_pose.Pose(static_image_mode=True) as pose:
        height, width, _ = image.shape

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        landmarks = result.pose_landmarks

        if result.pose_landmarks is not None:
            dict_landmarks = crear_dict(image, landmarks)

            medidas['shoulders'] = calcular_distancia(dict_landmarks['left_shoulder'], dict_landmarks['right_shoulder'])
            medidas['hip'] = calcular_distancia(dict_landmarks['left_hip'], dict_landmarks['right_hip'])
            
            left_dist = calcular_distancia(dict_landmarks['left_shoulder'], dict_landmarks['left_hip'])
            right_dist = calcular_distancia(dict_landmarks['right_shoulder'], dict_landmarks['right_hip'])
            dist = (left_dist + right_dist) / 2
            medidas['chest'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_shoulder'], dict_landmarks['left_elbow'])
            right_dist = calcular_distancia(dict_landmarks['right_shoulder'], dict_landmarks['right_elbow'])
            dist = (left_dist + right_dist) / 2
            medidas['forearms'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_elbow'], dict_landmarks['left_wrist'])
            right_dist = calcular_distancia(dict_landmarks['right_elbow'], dict_landmarks['right_wrist'])
            dist = (left_dist + right_dist) / 2
            medidas['arms'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_hip'], dict_landmarks['left_knee'])
            right_dist = calcular_distancia(dict_landmarks['right_hip'], dict_landmarks['right_knee'])
            dist = (left_dist + right_dist) / 2
            medidas['thighs'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_knee'], dict_landmarks['left_ankle'])
            right_dist = calcular_distancia(dict_landmarks['right_knee'], dict_landmarks['right_ankle'])
            dist = (left_dist + right_dist) / 2
            medidas['legs'] = dist

            medidas['ears'] = calcular_distancia(dict_landmarks['left_ear'], dict_landmarks['right_ear'])

            left_dist = calcular_distancia(dict_landmarks['left_heel'], dict_landmarks['left_foot_index'])
            right_dist = calcular_distancia(dict_landmarks['right_heel'], dict_landmarks['right_foot_index'])
            dist = (left_dist + right_dist) / 2
            medidas['foots'] = dist

        for m in medidas:
            medidas_cm[m] = medidas[m]*height_cm/height_pxl


        #mp_drawing.draw_landmarks(sombra, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        '''cv2.imshow("landmarks1", image)
        cv2.imshow("landmarks2", sombra)'''
        
        resultados_medidas = [medidas,medidas_cm,dict_landmarks]
        return resultados_medidas

def measures_front_view(image,landmarks):
    
    measures_front = {'shoulders':0,'hips':0}
    coords_front = {'shoulders':[],'hips':[]}

    rows, cols, _ = image.shape

    # **********************shoulders*************************
    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    x1 = landmarks['left_shoulder'][0]
    y1 = landmarks['left_shoulder'][1]
    x2 = landmarks['right_shoulder'][0]
    y2 = landmarks['right_shoulder'][1]

    #print(image[rows - 1, cols - 1])

    for y in range(rows):
        for x in range(cols):
            if not (x > x1 or x < x2):
                imageOut[y, x] = np.array([0, 0, 0])
            if not (y < y1 or y < y2):
                imageOut[y, x] = np.array([0, 0, 0])
            if np.array_equal(image[y, x], np.array([192, 192, 192])) == True:
                imageOut[y, x] = np.array([0, 0, 0])
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for y in range(rows):
        for x in range(cols):
            if np.array_equal(imageOut[y, x], np.array([0, 0, 0])) == False:
                if x1 == 0 and y1 == 0:
                    x1 = x
                    y1 = y
                    x2 = x
                    y2 = y
                else:
                    if x < x1:
                        x1 = x
                        y1 = y
                    if x > x2:
                        x2 = x
                        y2 = y

    
    measures_front['shoulders']= x2 - x1
    coords_front['shoulders']= [x1, x2, y2]

    """ imageOut = cv2.line(imageOut, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('shoulders', imageOut) """

    # **********************hips*************************
    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    

    y1 = landmarks['left_hip'][1]
    x1 = 0
    x2 = 0
    px1 = 0
    px2 = 0
    py = 0
    y2 = landmarks['right_hip'][1]
    cont = 0
    red = False
    dist = 0
    for y in range(rows):
        if cont<1:
            i = 0
            for x in range(cols):

                if not (y > y1 or y > y2):
                    imageOut[y, x] = np.array([0, 0, 0])
                if np.array_equal(imageOut[y, x], np.array([192, 192, 192])) == True:
                    imageOut[y, x] = np.array([0, 0, 0])
                if np.array_equal(imageOut[y, x], np.array([0,0,255])) == True:
                    red = True
                    i += 1
                else:
                    i = 0
                if i == 1:
                    if x1 == 0 and x2 == 0:
                        x1 = x
                        x2 = x
                    else:
                        x1 = x2
                        x2 = x
                        aux = dist
                        dist = x2 -x1
                        if dist > aux:
                            py = y
                            px1 = x1
                            px2 = x2
            if red:
                cont += 1
                
                
        else:

            for x in range(cols):
                if x < px1 or x > px2:
                    imageOut[y, x] = np.array([0, 0, 0])
                if np.array_equal(image[y, x], np.array([192, 192, 192])) == True:
                    imageOut[y, x] = np.array([0, 0, 0])
                

    """ imageOut = cv2.line(imageOut, (px1, py), (px2, py), (0, 255, 0), 3)

    cv2.imshow('hips', imageOut) """
    measures_front['hips']= px2 - px1
    coords_front['hips']= [px1, px2, py]

    return [measures_front,coords_front]

def measures_side_view(image,landmarks):
    rows, cols, _ = image.shape

    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    distancias = { 'shoulders':0,'chest':0,'navel':0}
    coords = {'shoulders': [], 'chest': [], 'navel': []}
    
    parte = ''
    if landmarks['left_heel'] > landmarks['left_foot_index']:

        y1 = landmarks['right_shoulder'][1]
        y2 = landmarks['right_hip'][1]
    else:

        y1 = landmarks['left_shoulder'][1]
        y2 = landmarks['left_hip'][1]
    #print('y1: ', y1)
    #print('y2: ', y2)

    for y in range(rows):
        x1 = 0
        x2 = 0
        for x in range(cols):
            if y < y1:
                imageOut[y, x] = np.array([0, 0, 0])
            if y > y2:
                imageOut[y, x] = np.array([0, 0, 0])
            if np.array_equal(image[y, x], np.array([192, 192, 192])) == True:
                imageOut[y, x] = np.array([0, 0, 0])
            if np.array_equal(image[y, x], np.array([0, 0, 255])) == True:
                if y == y1:
                    parte = 'shoulders'
                    if x1 == 0:
                        x1 = x
                    else:
                        x2 = x
                else:
                    if y > y1 and y < (((y2 - y1) / 2) + y1):
                        parte = 'chest'
                        if x1 == 0:
                            x1 = x
                        else:
                            x2 = x
                    else:
                        if y >= (((y2 - y1) / 2) + y1) and y < y2:
                            parte = 'navel'
                            if x1 == 0:
                                x1 = x
                            else:
                                x2 = x
                        else:
                            parte =''
        dist = x2 - x1
        if parte != '':
            if dist > distancias[parte]:
                coords[parte] = [x1, x2, y]
                distancias[parte] = dist

    
    for c in coords:
        x1 = coords[c][0]
        x2 = coords[c][1]
        y = coords[c][2]
        imageOut = cv2.line(imageOut, (x1,y), (x2,y), (0, 255, 0), 1)


    #cv2.imshow('shoulders(Lado)', imageOut)
    return [distancias,coords]

def calculate_measures_views(resp={'front':None,'side':None}):
    global measures_pxl
    global measures_cm
    front = resp['front'][0]
    side = resp['side'][0]
    result=resp['side'][0]

    """ print(front)
    print(side) """
    for r in side:
        if r == 'shoulders':
            a=front[r] /2
        else:
            a=front['hips'] /2
        b=side[r]/2
        h=((a-b)/(a+b))**2
        result[r]= round(3.1416 * (a+b)*(1+((3*h)/(10+((4-(3*h))**(1/2))))),2)
    #print(result)
    for r in result:
        measures_pxl[r]=result[r]
        measures_cm[r] = round(result[r]*height_cm/height_pxl,2)
    
    """ m_shoulder_front = front[0]['shoulders']
    m_hip_front = front[0]['hips']
    
    measures_pxl['shoulder'] = m_shoulder
    measures_pxl['hip'] = m_hip
    measures_cm['shoulder'] = round(m_shoulder*height_cm/height_pxl,2)
    measures_cm['hip'] = round(m_hip*height_cm/height_pxl,2) """

def draw_measures(image,coords,view):

    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    
    for coord in coords:
        x1 = coords[coord][0]
        x2 = coords[coord][1]
        y = coords[coord][2] 

        imageOut = cv2.line(imageOut, (x1,y), (x2,y), (0, 255, 0), 1)
    rute = "assets/out_"+view+".png"
    cv2.imwrite(rute, imageOut) #Imagen a mostrar
    


def measures():
    global measures_cm
    global measures_pxl
    global respImage
    images = {'front':None,'side':None}
    resp = {'front':None,'side':None}
    for view in images:
        path = 'person_'+view+'.png'
        images[view] = cv2.imread(path)
        rows, cols, _ = images[view].shape
        x = round(cols*600/rows)
        redim = cv2.resize(images[view], (x, 600))
        images[view] = redim

        seg = segmentation(images[view]) #[[output_image,dibujar_contornos(output_image,0)],[output_image, dibujar_contornos(output_image,1)]]
        if view == 'front':
            '''cv2.imshow('prueba1',seg[0][0])
            cv2.imshow('prueba2',seg[0][1])
            cv2.imshow('prueba3',seg[1][0])
            cv2.imshow('prueba4',seg[1][1])''' 
            def_height(seg[1][1])
            
            #**** medidas calculadas entre los puntos***
            medidas = postura(images[view],seg[1][0]) #[medidas,medidas_cm,dict_landmarks]
            #print('Medidas Otenidas:')
            for key in medidas[1]:
                m = medidas[1][key]
                medidas[1][key]=round(m,2)
                #print(key,": ",round(m,2)," cm")
            measures_cm = medidas[1]
            measures_pxl = medidas[0]
            

            #***medidas obtenidas entre los extremos (Front)***
            resp[view] = measures_front_view(seg[1][0],medidas[2])
           

        if view == 'side':
            '''cv2.imshow('prueba1',seg[0][0])
            cv2.imshow('prueba2',seg[0][1])
            cv2.imshow('prueba3',seg[1][0])
            cv2.imshow('prueba4',seg[1][1])''' 
            def_height(seg[1][1])
            
            #**** medidas calculadas entre los puntos***
            medidas = postura(images[view],seg[1][0]) #[medidas,medidas_cm,dict_landmarks]
            #print('Medidas Otenidas:')
            for key in medidas[1]:
                m = medidas[1][key]
                medidas[1][key]=round(m,2)
                #print(key,": ",round(m,2)," cm")
            
            measures_cm['foots'] = medidas[1]['foots']
            measures_pxl['foots'] = medidas[0]['foots']
            #***medidas obtenidas entre los extremos (Side)***
            resp[view] = measures_side_view(seg[1][0],medidas[2])
        draw_measures(seg[1][0],resp[view][1],view)

    #***medidas calculadas entre Front y Side***
    calculate_measures_views(resp)

    #***dibujar las medidas****
    imageFront = cv2.imread('assets/out_front.png') 
    imageSide = cv2.imread('assets/out_side.png')

    #Concatenar ambas imagenes
    imageOut = cv2.hconcat([imageFront,imageSide])
    rute = "assets/out.png"
    cv2.imwrite(rute, imageOut) #Imagen a mostrar
    imgNew = open(rute, 'rb') 
    image_read = imgNew.read()
    image_64_encode = base64.b64encode(image_read)
    respImage['base'] = str(image_64_encode)
    #print(image)
    
    """ cv2.waitKey(0)
    cv2.destroyAllWindows() """


if __name__ == '__main__':
    app.run(debug=DEBUG, port=PORT)
    CORS(app, resources={r"*": {"origins": "*"}},support_credentials=True)