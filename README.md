## Calculate measures

### Info:
PORT: 8000
after_request(): Habilitacion de CORS
not_found(): Error 404

### Requeriments:
- Pillow,
- Flask,
- Flask_cors,
- opencv-contrib-python o opencv-python-headless,
- mediapipe

### Usabilidad:
- 1. define_images()
    - view: 'Front o Side'
    - image: 'Nombre de la image'
    - base: 'Image en base64(con cabeceras)' (luego se maneja el string para eliminar la cabecera)
- 2. calculate_measure(): verifica que existan ambas imagenes, llama la funcion measures() y retorna la imaegn y medidas generadas
### Funciones:
- def_height(image): Recive una imagen de solo bordes para calcular la altura entre el punto mas alto y mas bajo dibujado  
- calcular_distancia(p1, p2): Calcla la distancia entre p1 y p2 (p1 y p2 son arrays con coordenadas [x,y])
- segmentation(img): Para eliminar el fondo y filtrar solo a la persona
- crear_dict(img, landmarks): Generar un diccionario de coordenadas de los puntos de las partes relevantes del cuerpo
- postura(image): Orquesta las funciones de crear_dict con calcular_distacia y genera distancias en cm y pxl  
- measures_front_view(image,landmarks): 
- measures_side_view(image,landmarks)
- calculate_measures_views(resp={'front':None,'side':None})
- draw_measures(image,coords,view)
- measures(): Gestionar las funciones para generar las medidas
