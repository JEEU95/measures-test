import cv2

import mediapipe as mp
import numpy as np
#import tensorflow as tf
#from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import matplotlib.pyplot as plt

#bodypix_model=load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
altura_cm = 172
altura_pxl = 0
'''tranformar los puntos detectados en el cuerpo a un diccionario de coordenadas x,y'''

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


'''Calcular la distancia entre dos puntos en pixels'''


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
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        output_image = np.where(condition, image, bg_image)

        cv2.imshow('mask_output0', output_image)
        cv2.imwrite('assets/out.png',output_image)
        mask = cv2.medianBlur(output_image, 13)
        cv2.imshow('mask', mask)

        results_mask.append([output_image,dibujar_contornos(output_image,0)])

        output_image = np.where(condition, fg_image, bg_image)
        cv2.imshow('mask_output1', output_image)

        results_mask.append([output_image, dibujar_contornos(output_image,1)])

        return results_mask

def dibujar_contornos(image,n):
    bordeCanny = cv2.Canny(image,100,100)
    ctns, _ = cv2.findContours(bordeCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, ctns, -1, (0, 0, 255), 2)

    bordes = cv2.bitwise_not(bordeCanny)
    archivo='assets/bordes'+str(n)+'.jpg'

    cv2.imwrite(archivo,bordes)
    #cv2.imshow('canny'+str(n),image)
    return bordes

def trazar_limites(image,p1,p2):
    rows, cols = image.shape
    x1=0
    x2=400
    x = abs(p2[0] - p1[0])
    y = abs(p2[1] - p1[1])
    ang = np.arctan(y/x)
    if p2[0] > p1[0]:
        y1 = p1[1]-abs(p1[0] - x1) * np.tan(ang)
        y2 = abs(x2 - p2[0]) * np.tan(ang) +p2[1]

    else:
        y1 = p2[1]-abs(p2[0] - x1) * np.tan(ang)
        y2 = abs(x2 - p1[0]) * np.tan(ang)+p1[1]
    y1 = round(y1)
    y2 = round(y2)

    c1 = []
    c2 = []
    for c in range(cols):
        if image[y1, c] == 0:
            if c1 == []:
                c1 = [c, y1]
                break
    for c in range (cols):
        if image[y2,c] == 0:
            c2 = [c,y1]

    image = cv2.line(image,(x1,y1),(x2,y2),(0,0,0),1)

    print(cols)
    '''
    cv2.circle(image,(c1[0],c1[1]),3,(255,255,255),3)
    cv2.circle(image,(c2[0],c2[1]),3,(255,255,255),3)

    '''
    #cv2.imshow("limites", image)
    return image
def postura(image,sombra):
    medidas = {'altura': altura_pxl,
               'hombros': 0,
               'cadera': 0,
               'pecho': 0,
               'brazos': 0,
               'antebrazos': 0,
               'piernas': 0,
               'muslos': 0,
               'orejas': 0,
               'pies': 0
               }
    medidas_cm = {'altura': altura_cm,
               'hombros': 0,
               'cadera': 0,
               'pecho': 0,
               'brazos': 0,
               'antebrazos': 0,
               'piernas': 0,
               'muslos': 0,
               'orejas': 0,
               'pies': 0
               }
    with mp_pose.Pose(static_image_mode=True) as pose:
        height, width, _ = image.shape

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        landmarks = result.pose_landmarks

        print("----Pose Landmarks----")
        print(landmarks)

        if result.pose_landmarks is not None:
            dict_landmarks = crear_dict(img, landmarks)

            medidas['hombros'] = calcular_distancia(dict_landmarks['left_shoulder'], dict_landmarks['right_shoulder'])
            medidas['cadera'] = calcular_distancia(dict_landmarks['left_hip'], dict_landmarks['right_hip'])
            #sombra = trazar_limites(sombra,dict_landmarks['left_hip'], dict_landmarks['right_hip'])
            #sombra = trazar_limites(sombra,dict_landmarks['left_shoulder'], dict_landmarks['right_shoulder'])

            left_dist = calcular_distancia(dict_landmarks['left_shoulder'], dict_landmarks['left_hip'])
            right_dist = calcular_distancia(dict_landmarks['right_shoulder'], dict_landmarks['right_hip'])
            dist = (left_dist + right_dist) / 2
            medidas['pecho'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_shoulder'], dict_landmarks['left_elbow'])
            right_dist = calcular_distancia(dict_landmarks['right_shoulder'], dict_landmarks['right_elbow'])
            dist = (left_dist + right_dist) / 2
            medidas['antebrazos'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_elbow'], dict_landmarks['left_wrist'])
            right_dist = calcular_distancia(dict_landmarks['right_elbow'], dict_landmarks['right_wrist'])
            dist = (left_dist + right_dist) / 2
            medidas['brazos'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_hip'], dict_landmarks['left_knee'])
            right_dist = calcular_distancia(dict_landmarks['right_hip'], dict_landmarks['right_knee'])
            dist = (left_dist + right_dist) / 2
            medidas['muslos'] = dist

            left_dist = calcular_distancia(dict_landmarks['left_knee'], dict_landmarks['left_ankle'])
            right_dist = calcular_distancia(dict_landmarks['right_knee'], dict_landmarks['right_ankle'])
            dist = (left_dist + right_dist) / 2
            medidas['piernas'] = dist

            medidas['orejas'] = calcular_distancia(dict_landmarks['left_ear'], dict_landmarks['right_ear'])

            left_dist = calcular_distancia(dict_landmarks['left_heel'], dict_landmarks['left_foot_index'])
            right_dist = calcular_distancia(dict_landmarks['right_heel'], dict_landmarks['right_foot_index'])
            dist = (left_dist + right_dist) / 2
            medidas['pies'] = dist

        for m in medidas:
            medidas_cm[m] = medidas[m]*altura_cm/altura_pxl


        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow("image2", image)
        resultados_medidas = [medidas,medidas_cm,dict_landmarks]
        return resultados_medidas

def def_altura(image):
    global altura_pxl
    print(altura_pxl)
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
    altura = y2 - y1
    altura_pxl = altura
    color = (0, 255, 0)
    img = cv2.imread("assets/bordes1.jpg")

    img = cv2.line(img, (x1, y1), (x2, y2), color, 3)
    #cv2.imshow('altura', img)

def dibujar_medidas(image,medidas,landmarks):

    rows, cols, _ = image.shape

    # **********************Medidas-Lado*************************
    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)

    distancias = { 'hombros':0,'pecho':0,'ombligo':0}
    coords = {'hombros': [], 'pecho': [], 'ombligo': []}
    parte = ''
    if landmarks['left_heel'] > landmarks['left_foot_index']:

        y1 = landmarks['right_shoulder'][1]
        y2 = landmarks['right_hip'][1]
    else:

        y1 = landmarks['left_shoulder'][1]
        y2 = landmarks['left_hip'][1]
    print('y1: ', y1)
    print('y2: ', y2)

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
                    parte = 'hombros'
                    if x1 == 0:
                        x1 = x
                    else:
                        x2 = x
                else:
                    if y < (((y2 - y1) / 2) + y1):
                        parte = 'pecho'
                        if x1 == 0:
                            x1 = x
                        else:
                            x2 = x
                    else:
                        if y > (((y2 - y1) / 2) + y1)+((y2 - (((y2 - y1) / 2) + y1))/4) and y < y2 - ((y2 - (((y2 - y1) / 2) + y1))/4):
                            parte = 'ombligo'
                            if x1 == 0:
                                x1 = x
                            else:
                                x2 = x
        dist = x2 - x1
        if parte != '':
            if dist > distancias[parte]:
                coords[parte] = [x1, x2, y]
                distancias[parte] = dist

    print(coords)
    for c in coords:
        x1 = coords[c][0]
        x2 = coords[c][1]
        y = coords[c][2]
        imageOut = cv2.line(imageOut, (x1,y), (x2,y), (0, 255, 0), 1)


    cv2.imshow('hombros(Lado)', imageOut)

    for d in distancias:
        print(d,': ', distancias[d])


    '''# **********************Hombros*************************
    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)
    x1 = landmarks['left_shoulder'][0]
    y1 = landmarks['left_shoulder'][1]
    x2 = landmarks['right_shoulder'][0]
    y2 = landmarks['right_shoulder'][1]

    print(image[rows - 1, cols - 1])

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

    print('x1: ', x1, ' - x2: ', x2)
    print('y1: ', y1, ' - y2: ', y2)
    print('Diametro1(hombros): ', x2 - x1)

    imageOut = cv2.line(imageOut, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('hombros', imageOut)

    # **********************Caderas*************************
    imageOut = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_HSV2BGR)

    y1 = landmarks['left_hip'][1]
    x1 = 0
    x2 = 0
    distancias = []
    posX = []
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
                if np.array_equal(imageOut[y, x], np.array([192, 192, 192])) == True:
                    imageOut[y, x] = np.array([0, 0, 0])

    for x in range(cols):
        if x < px1 or x > px2:
            imageOut[py, x] = np.array([0, 0, 0])
        if np.array_equal(imageOut[y, x], np.array([192, 192, 192])) == True:
            imageOut[py, x] = np.array([0, 0, 0])
    imageOut = cv2.line(imageOut, (px1, py), (px2, py), (0, 255, 0), 1)

    cv2.imshow('caderas', imageOut)
    '''



def segment_color(image, part=''):
    segments = ['head','chest','legs','arms', 'feet']
    colorBajo1 = np.array([80, 210, 164], np.uint8)
    colorAlto1 = np.array([115, 219, 182], np.uint8)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imageHSV, colorBajo1, colorAlto1)
    mask = cv2.medianBlur(mask, 13)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    #cv2.imshow('segment-color',mask)

if __name__ == '__main__':
    ruta="assets/img1.jpg"
    img = cv2.imread(ruta)
    rows, cols,z = img.shape
    x=round(cols*600/rows)
    #altura_cm = int(input('Indique su altura en cm: '))

    redim = cv2.resize(img, (x, 600))
    img = redim

    #kernel = np.ones((5,5),np.float32)/25
    #img = cv2.filter2D(img,-1,kernel)


    #cv2.imshow("image", img)

    #cv2.imshow('redim',redim)
    seg = segmentation(img)
    def_altura(seg[1][1])
    medidas = postura(img,seg[1][1])

    print('Medidas Otenidas:')
    for key in medidas[1]:
        m = medidas[1][key]
        print(key,": ",round(m,2)," cm")

    dibujar_medidas(seg[1][0],medidas[1],medidas[2])

    rutaNueva = "assets/out.png"
    imgNueva = cv2.imread(rutaNueva)
    '''
    res = bodypix_model.predict_single(imgNueva)
    mask = res.get_mask(threshold=0.5).numpy().astype(np.uint8)
    # colored mask (separate colour for each body part)
    colored_mask = res.get_colored_part_mask(mask)
    # colored mask (separate colour for each body part)
    colored_mask = res.get_colored_part_mask(mask)
    ruta = "assets/colored-mask.png"
    tf.keras.preprocessing.image.save_img(ruta,colored_mask)
    color = cv2.imread(ruta)
    cv2.imshow('body-pix',color)
    segment_color(color)
    '''
    cv2.waitKey(0)

cv2.destroyAllWindows()

'''
***mask mediapipe segmentation intersect with mediapipe landsmarks
***define this segmentation
***the green color in the segmentation mask if the clothing measurement is correct or red otherwise
'''