from distutils.debug import DEBUG
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

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

measures_cm = {'Bust' :0,'Waist':0,'Hip':0,'Neck':0,'Shoulder':0,'Arm':0,'Chest':0,'height':0,'Inseam':0,'Leg':0, 'Bicep':0,'Thigh':0}
measures_front={}
measures_side={}

landmarks = {}

""" measures_cm = {'neck':0,'Shoulders': 0,'Arms': 0,'Chest': 0,'hips': 0,'Legs': 0,'Feet': 0}
measures_pxl = {'neck':0, 'Shoulders': 0,'Arms': 0,'Chest': 0,'hips': 0,'Legs': 0,'Feet': 0}
measures_front= {'height': 0,'shoulders': 0,'hip': 0,'chest': 0,'arms': 0,'forearms': 0,'legs': 0,'thighs': 0,'neck': 0,'foots': 0}
measures_side= {'height': 0,'shoulders': 0,'hip': 0,'chest': 0,'arms': 0,'forearms': 0,'legs': 0,'thighs': 0,'neck': 0,'foots': 0}
 """

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
    #print(request_measure) #ver el json a procesar

    if 'name' in request_measure and 'measure_cm' in request_measure: #Cambiar del front y back el parametro 'name' por 'param'
        if request_measure['name'] == 'height':
            height_cm = request_measure['measure_cm']            
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
    #print(request_img) #ver el json a procesar(contiene la img en)
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


def def_height(view,image=None):
    global height_pxl
    #cv2.imshow('image', image)
    #print(height_pxl)
    #if view == 'front' :
    if view == 'front':
        nose = landmarks['nose'][1]
        eyeR = landmarks['right_eye'][1]
        eyeL = landmarks['left_eye'][1]
        eyeM = int((eyeL+eyeR) / 2)
        partEN = nose - eyeM
        h4 = nose - partEN*4
        h5 = nose - partEN*5
        hM = int((h4 + h5) /2) #Punto calculado de la nuca

        hipR = landmarks['right_hip'][1]
        hipL = landmarks['left_hip'][1]
        hipM = int((hipR+hipL) / 2)

        d1 = hipM - hM #distancia de la nuca a la cadera

        daR = distance_points(landmarks['right_hip'], landmarks['right_knee'])
        dbR = distance_points(landmarks['right_knee'], landmarks['right_heel'])
        d2R = daR + dbR #Medida de la pierna derecha

        daL = distance_points(landmarks['left_hip'], landmarks['left_knee'])
        dbL = distance_points(landmarks['left_knee'], landmarks['left_heel'])
        d2L = daL + dbL #Medida de la pierna izquierda

        d2 = (d2R + d2L) / 2 #Media de la medida de las piernas

        height_pxl = d1 + d2 #Pixeles calculados  equivalentes a la altura ingresada
        
    else:
        if landmarks['left_heel'][0] > landmarks['left_foot_index'][0]:
            view = 'left'
        else:
            view = 'right'
        nose = landmarks['nose'][1]
        eye = landmarks[view+'_eye'][1]
        partEN = nose - eye
        h4 = nose - partEN*4
        h5 = nose - partEN*5
        hM = int((h4 + h5) /2) #Punto calculado de la nuca

        hip = landmarks[view+'_hip'][1]
        
        d1 = hip - hM #distancia de la nuca a la cadera

        da = distance_points(landmarks[view+'_hip'], landmarks[view+'_knee'])
        db = distance_points(landmarks[view+'_knee'], landmarks[view+'_heel'])
        d2 = da + db 

        height_pxl = d1 + d2 #Pixeles calculados  equivalentes a la altura ingresada
    

    """ x = landmarks['nose'][0]
    y1 = round(hM)
    y2 = round(y1+height_pxl)
    rows, cols = image.shape
  
    color = (0, 255, 0)
    img = cv2.imread("assets/bordes1.jpg")
    img = cv2.line(img, (x, y1), (x, y2), (0,0,255), 3)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  """

def distance_points(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

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

def create_landmarks(img, land):
    global landmarks
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
        landmarks[var] = [int(land.landmark[i].x * width), int(land.landmark[i].y * height)]

def define_posture(image,sombra,view):
    
    with mp_pose.Pose(static_image_mode=True) as pose:
        height, width, _ = image.shape
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        land = result.pose_landmarks

        if result.pose_landmarks is not None:
            #Calcular la distancia entre cada interseccion
            create_landmarks(image, land)
            def_height(view,sombra)
            """ 
            if view == 'front':
            #if view front/left/right

                medidas['shoulders'] = distance_points(dict_landmarks['left_shoulder'], dict_landmarks['right_shoulder'])
                medidas['hip'] = distance_points(dict_landmarks['left_hip'], dict_landmarks['right_hip'])
                
                left_dist = distance_points(dict_landmarks['left_shoulder'], dict_landmarks['left_hip'])
                right_dist = distance_points(dict_landmarks['right_shoulder'], dict_landmarks['right_hip'])
                dist = (left_dist + right_dist) / 2
                medidas['chest'] = dist

                left_dist = distance_points(dict_landmarks['left_shoulder'], dict_landmarks['left_elbow'])
                right_dist = distance_points(dict_landmarks['right_shoulder'], dict_landmarks['right_elbow'])
                dist = (left_dist + right_dist) / 2
                medidas['forearms'] = dist

                left_dist = distance_points(dict_landmarks['left_elbow'], dict_landmarks['left_wrist'])
                right_dist = distance_points(dict_landmarks['right_elbow'], dict_landmarks['right_wrist'])
                dist = (left_dist + right_dist) / 2
                medidas['arms'] = dist

                left_dist = distance_points(dict_landmarks['left_hip'], dict_landmarks['left_knee'])
                right_dist = distance_points(dict_landmarks['right_hip'], dict_landmarks['right_knee'])
                dist = (left_dist + right_dist) / 2
                medidas['thighs'] = dist

                left_dist = distance_points(dict_landmarks['left_knee'], dict_landmarks['left_ankle'])
                right_dist = distance_points(dict_landmarks['right_knee'], dict_landmarks['right_ankle'])
                dist = (left_dist + right_dist) / 2
                medidas['legs'] = dist

                medidas['neck'] = (distance_points(dict_landmarks['left_ear'], dict_landmarks['right_ear']))*3.1416
            else:
                if dict_landmarks['left_heel'][0] > dict_landmarks['left_foot_index'][0]:
                    view = 'left'
                else:
                    view = 'right'
                
                dist = distance_points(dict_landmarks[view+'_shoulder'], dict_landmarks[view+'_hip'])
                medidas['chest'] = dist

                dist = distance_points(dict_landmarks[view+'_shoulder'], dict_landmarks[view+'_elbow'])
                dist = (left_dist + right_dist) / 2
                medidas['forearms'] = dist

                dist = distance_points(dict_landmarks[view+'_elbow'], dict_landmarks[view+'_wrist'])
                dist = (left_dist + right_dist) / 2
                medidas['arms'] = dist

                dist = distance_points(dict_landmarks[view+'_hip'], dict_landmarks[view+'_knee'])
                dist = (left_dist + right_dist) / 2
                medidas['thighs'] = dist

                dist = distance_points(dict_landmarks[view+'_knee'], dict_landmarks[view+'_ankle'])
                dist = (left_dist + right_dist) / 2
                medidas['legs'] = dist

                medidas['neck'] = (distance_points(dict_landmarks[view+'_ear'], dict_landmarks[view+'__ear']))*3.1416 """
            

            """ left_dist = distance_points(dict_landmarks['left_heel'], dict_landmarks['left_foot_index'])
            right_dist = distance_points(dict_landmarks['right_heel'], dict_landmarks['right_foot_index'])
            dist = (left_dist + right_dist) / 2
            medidas['foots'] = dist """

        
        """ medidas['height'] = height_pxl
        measures_cm['height'] = height_cm
        
        
        for m in medidas:
            measureC=medidas[m]*height_cm/height_pxl
            medidas_cm[m] = round(measureC,2)


        #mp_drawing.draw_landmarks(sombra, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        '''cv2.imshow("landmarks1", image)
        cv2.imshow("landmarks2", sombra)'''
        
        resultados_medidas = [medidas,medidas_cm,dict_landmarks]
        return resultados_medidas """

def measures_front_view(image):
    # ******* Todas las medidas *******
    global measures_front
    measures_front={'Bust' :0,'Waist':0,'Hip':0,'Neck':0,'Shoulder':0,'Arm':0,'Chest':0,'height':0,'Inseam':0,'Leg':0,'Bicep':0,'Thigh':0} 
    coords_front = {}#coordenadas para dibujar
    rows, cols, _ = image.shape


    # *************************Height*************************
    measures_front['height'] = height_pxl


    # *************************Bust*************************
    measures_front['Bust']= distance_points(landmarks['left_shoulder'], landmarks['right_shoulder'])/2

    
    # ************************Waist************************
    # *Requeriment* The person must pose with their hands on their waist.
    # The minimum distance betewen the hands
    waist1=distance_points(landmarks['left_hip'], landmarks['right_hip'])*2  #tomando la distancia entre los puntos de la cintura *2
    
    xRHand1,_ = landmarks['right_wrist']
    xRHand2,_ = landmarks['right_pinky']
    xRHand3,_ = landmarks['right_index']
    xRHand4,_ = landmarks['right_thumb']

    right_hand = [max([xRHand1,xRHand2,xRHand3,xRHand4]),landmarks['right_hip'][1]]
    
    xLHand1,_ = landmarks['left_wrist']
    xLHand2,_ = landmarks['left_pinky']
    xLHand3,_ = landmarks['left_index']
    xLHand4,_ = landmarks['left_thumb']
    left_hand = [min([xLHand1,xLHand2,xLHand3,xLHand4]),landmarks['left_hip'][1]]
    
    waist2=distance_points(right_hand,left_hand) #distancia entre los puntos mas cercanos de cada mano a la cadera
    
    waist3=(waist1+waist2)/2 #media entre ambos calculos

    measures_front['Waist']= min([waist1,waist2,waist3])/2


    # ************************Neck************************
    measures_front['Neck'] = distance_points(landmarks['right_ear'], landmarks['left_ear'])/2
    
    
    # ************************Arms************************
    right_arm = distance_points(landmarks['right_shoulder'], landmarks['right_elbow'])+ distance_points(landmarks['right_elbow'], landmarks['right_wrist'])
    left_arm= distance_points(landmarks['left_shoulder'], landmarks['left_elbow']) + distance_points(landmarks['left_elbow'], landmarks['left_wrist'])
    measures_front['Arm'] = (right_arm + left_arm)/2

    
    # ************************Chest (Height)************************
    _,yRShoulder =landmarks['right_shoulder']
    _,yLShoulder =landmarks['left_shoulder']
    _,yRHip =landmarks['right_hip']
    _,yLHip =landmarks['left_hip']
    yShoulder =(yRShoulder+yLShoulder)/2
    yHip = (yRHip+yLHip)/2
    measures_front['Chest']=abs(yHip-yShoulder)

    # ************************Thigh************************
    right_thigh = distance_points(landmarks['right_hip'],landmarks['right_knee'])
    left_thigh = distance_points(landmarks['left_hip'],landmarks['left_knee'])
    measures_front['Thigh']=(right_thigh+left_thigh)/2

    # ************************Legs************************
    """ left_ankle = distance_points(landmarks['left_knee'], landmarks['left_heel'])
    right_ankle = distance_points(landmarks['right_knee'], landmarks['right_heel'])
    leg1 = (right_ankle+left_ankle) / 2
    leg2 = measures_front['Thigh']
    leg=leg1+leg2 """
    leg1=distance_points(landmarks['right_hip'], landmarks['right_heel'])
    leg2=distance_points(landmarks['left_hip'], landmarks['left_heel'])
    
    leg=(leg1+leg2)/2
    measures_front['Leg']=leg
   
    
    # ************************Iseam************************
    iseam = abs(measures_front['Thigh']/3)
    yIseam = yHip+iseam
    yLeg = yHip+leg

    _,yRHeel = landmarks['right_heel']
    _,yLHeel = landmarks['left_heel']

    measures_front['Inseam'] = yLeg-yIseam

    # ************************Biceps************************
    shoulder = landmarks['right_shoulder']
    elbow = landmarks['right_elbow']
    
    y = abs(elbow[1] - shoulder[1])
    x = abs(elbow[0] - shoulder[0])
    z = distance_points(shoulder,elbow)
    
    a = np.degrees(np.arctan(y/x))
    b = np.degrees(np.arctan(x/y))
    B = a + 90
    A = 180 - B

    dist = z / 2

    x1 = round(elbow[0]+ (dist * np.cos(np.radians(a))))
    y1 = round(elbow[1]-(dist * np.sin(np.radians(a))))

    imageOut = image.copy()
    for i in range(x1):
        x2 = x1 - i
        y2 = y1 - round(x2 * np.tan(np.radians(A)))
        if y2 >=0 :
            if np.array_equal(image[y2, i], np.array([0,0,255])) == True:
                measures_front['Bicep']=distance_points([i, y2], [x1, y1])
                coords_front['Bicep']=[i,x1, y2, y1]
                """ print([i, y2])
                print([x1, y1]) """
                
                """ imageOut = cv2.line(imageOut, (i, y2), (x1, y1), (0, 255, 0), 1)
                imageOut = cv2.line(imageOut, (elbow[0],elbow[1]), (shoulder[0],shoulder[1]), (255, 0, 0), 1)
                cv2.imshow('medidas', imageOut) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()  """
                break

    # ************************Shoulders************************
    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    x1,y1 = landmarks['left_shoulder']
    x2,y2 = landmarks['right_shoulder']

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

    neck = measures_front['Neck']*2
    dist_shoulders = x2-x1
    measures_front['Shoulder']= (dist_shoulders-neck)/2
    coords_front['shoulders']= [x1, x2, y2]

    """ imageOut = cv2.line(imageOut, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('shoulders', imageOut) """
    """ 
    # **********************hips*************************
    # Cambiar a waist
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
    """
    """ imageOut = cv2.line(imageOut, (px1, py), (px2, py), (0, 255, 0), 3)

    cv2.imshow('hips', imageOut) """
    """ measures_front['Waist']= px2 - px1
    coords_front['hips']= [px1, px2, py] """

    measures_front=transform_pxl2cm(measures_front)

    return coords_front

def measures_side_view(image):

    global measures_side
    measures_side = {'Bust' :0,'Waist':0,'Hip':0,'Neck':0,'Bicep':0,'Thigh':0}

    rows, cols, _ = image.shape

    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    coords = {}
    
    parte = ''
    if landmarks['left_heel'][0] > landmarks['left_foot_index'][0]:
        _,yEar=landmarks['right_ear']
        _,yShoulder =landmarks['right_shoulder']
        _,yKnee = landmarks['right_knee']
        _,yHip = landmarks['right_hip']
    else:
        _,yEar=landmarks['left_ear']
        _,yShoulder =landmarks['left_shoulder']
        _,yKnee = landmarks['left_knee']
        _,yHip = landmarks['left_hip']
        
    yNeck =range(yEar,yShoulder)  
    yBust = round(((yHip-yShoulder)/3)+yShoulder)
    yWaist= round((2*(yHip-yShoulder)/3)+yShoulder)
    yThigh= round((yKnee+yHip)/2)

    for y in [yBust,yWaist,yHip,yNeck,yThigh]:
        if type(y) != int:
            dist = 0
            for j in y:
                x1 = 0
                x2 = 0
                for x in range(cols):
                    if np.array_equal(image[j, x], np.array([0, 0, 255])) == True:
                        if x1 == 0:
                            x1 = x
                        else:
                            x2 = x
                        
                if dist == 0:
                    dist = x2-x1
                    coords['Neck'] = [x1, x2, j]
                    measures_side['Neck'] = dist/2
                else:
                    aux = x2-x1
                    if dist>aux:
                        dist = aux         
                        coords['Neck'] = [x1, x2, j]
                        measures_side['Neck'] = dist/2
        else:
            x1 = 0
            x2 = 0
            for x in range(cols):   
                if np.array_equal(image[y, x], np.array([0, 0, 255])) == True:
                    parte ='none'
                    if y == yBust:
                        parte = 'Bust'
                        if x1 == 0:
                            x1 = x
                        else:
                            x2 = x
                    if y == yWaist:
                        parte = 'Waist'
                        if x1 == 0:
                            x1 = x
                        else:
                            x2 = x
                    if y == yHip:
                        parte = 'Hip'
                        if x1 == 0:
                            x1 = x
                        else:
                            x2 = x
                    if y == yThigh:
                        parte = 'Thigh'
                        if x1 == 0:
                            x1 = x
                        else:
                            x2 = x
                    
            if parte !='none':        
                dist = x2 - x1           
                measures_side[parte] = dist/2
                coords[parte] = [x1, x2, y]    

    """ for c in coords:
        x1,x2,y = coords[c]
        imageOut = cv2.line(imageOut, (x1,y), (x2,y), (0, 255, 0), 1) """

    #cv2.imshow('shoulders(Lado)', imageOut)
    measures_side = transform_pxl2cm(measures_side)
    return coords
def transform_pxl2cm(measures):
    pxl = height_pxl
    cm = height_cm

    for key in measures:
        m = measures[key]
        measures[key]=round((m*cm/pxl),2)
    
    return measures

def calculate_measures_views():
    global measures_cm
    front = measures_front
    side = measures_side

    side['Bicep']=front['Bicep']/2
    front['Thigh']=side['Thigh']/2

    result = front
    
    for key in side:
        a = front[key]
        b = side[key]

        h=((a-b)/(a+b))**2
        z=3*h   

        res=round(3.1416*(a+b)*(1 +(z / (10+((4-z)**(1/2))))),2)

        result[key]=res
    
    measures_cm=result


def draw_measures(image,coords,view):
    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    xcut1=0
    ycut1=0
    xcut2=0
    ycut2=0

    for coord in coords:
        if len(coords[coord])==3:
            x1,x2,y = coords[coord]
            imageOut = cv2.line(imageOut, (x1,y), (x2,y), (0, 255, 0), 1)
        if len(coords[coord])==4:
            x1,x2,y1,y2 = coords[coord]
            imageOut = cv2.line(imageOut, (x1,y1), (x2,y2), (0, 255, 0), 1)

    rute = "assets/out_"+view+".png"
    rows, cols, _ = imageOut.shape
    for y in range(rows):
        for x in range(cols):
            if np.array_equal(image[y, x], np.array([0,0,255])) == True:
                if xcut1 == 0 and ycut1 == 0:
                    xcut1=x
                    ycut1=y
                else:
                    if x < xcut1:
                        xcut1=x
                    if x > xcut2:
                        xcut2 = x
                    if y < ycut1:
                        ycut1=y
                    if y > ycut2:
                        ycut2 = y
    if ycut1>0:
        ycut1=ycut1-1
    if xcut1>0:
        xcut1=xcut1-1
    ycut2=ycut2+1
    xcut2=xcut2+1
    imageOut = imageOut[ycut1:ycut2,xcut1:xcut2]
    
    cv2.imwrite(rute, imageOut) #Imagen a mostrar

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def measures():
    global measures_cm
    global respImage
    try:
        os.mkdir('assets')
    except OSError as e:
        print('')
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
            '''cv2.imshow('prueba1F',seg[0][0])
            cv2.imshow('prueba2F',seg[0][1])
            cv2.imshow('prueba3F',seg[1][0])
            cv2.imshow('prueba4F',seg[1][1]) '''
            
            #**** medidas calculadas entre los puntos***
            define_posture(images[view],seg[1][1],view)            

            #***coords y actualizacion de las medidas obtenidas entre los extremos (Front)***
            resp[view] = measures_front_view(seg[1][0])
            draw_measures(seg[1][0],resp[view],view)  #Para dibujar las medidas obtenidas y dibuar las imagenes a retornar
           

        if view == 'side':
            '''cv2.imshow('prueba1S',seg[0][0])
            cv2.imshow('prueba2S',seg[0][1])
            cv2.imshow('prueba3S',seg[1][0])
            cv2.imshow('prueba4S',seg[1][1])''' 
            
            #**** medidas calculadas entre los puntos***
            define_posture(images[view],seg[1][1],view)
            
            #***coords y actualizacion de las medidas obtenidas entre los extremos (Side)***
            resp[view] = measures_side_view(seg[1][0])
            draw_measures(seg[1][0],resp[view],view)  #Para dibujar las medidas obtenidas y dibuar las imagenes a retornar 

    #***medidas calculadas entre Front y Side***
    calculate_measures_views()

    #***dibujar las medidas****
    imageFront = cv2.imread('assets/out_front.png') 
    imageSide = cv2.imread('assets/out_side.png')

    #Concatenar ambas imagenes   
    rute = "assets/out.png"
    imageOut = hconcat_resize_min([imageFront,imageSide])#imageOut = cv2.hconcat([imageFront,imageSide])
    cv2.imwrite(rute, imageOut) #Imagen a mostrar
    imgNew = open(rute, 'rb') 
    image_read = imgNew.read()
    image_64_encode = base64.b64encode(image_read)
    respImage['base'] = str(image_64_encode)
    imgNew.close()

    paths=['assets/bordes0.jpg','assets/bordes1.jpg','assets/out_front.png','assets/out_side.png','assets/out.png','person_front.png','person_side.png']
    for p in paths:
        os.remove(p)
    os.rmdir('assets')
    #print(image)

    """cv2.waitKey(0)
    cv2.destroyAllWindows() """

if __name__ == '__main__':
    app.run(debug=DEBUG, port=PORT)
    CORS(app, resources={r"*": {"origins": "*"}},support_credentials=True)