import numpy as np
import cv2
from matplotlib import pyplot as plt

#Funcion que solo deja las sombras del GT
def sombras(I):
   vert,hor=I.shape
   for i in range(vert):
      for j in range (hor):
         if(I[i,j]>=180):
            I[i,j]=0
         
         
#Funcion que calcula los Verdaderos Negativos entre GT y los resultados del algoritmo
def TrueNegatives(I,GT):
   q,w=I.shape
   acumulador=0
   for i in range(q):
      for j in range(w):
         if(I[i,j]==0 and GT[i,j]==0):
            acumulador+=1
   return   acumulador
   
#Funcion que calcula los Falsos Negativos entre GT y los resultados del algoritmo  
def FalseNegatives(I,GT):
   q,w=I.shape
   acumulador=0
   for i in range(q):
      for j in range(w):
         if(I[i,j]==0 and GT[i,j]!=0):
            acumulador+=1
   return   acumulador 
      
      
#Funcion que calcula los Falsos Positivos entre GT y los resultados del algoritmo
def FalsePositives(I,GT):
   q,w=I.shape
   acumulador=0
   for i in range(q):
      for j in range(w):
         if(I[i,j]!=0 and GT[i,j]==0):
            acumulador+=1
   return   acumulador


#Funcion que calcula los Verdaderos Positivos entre GT y los resultados del algoritmo
def TruePositives(I,GT):
   q,w=I.shape
   acumulador=0
   for i in range(q):
      for j in range(w):
         if(I[i,j]!=0 and GT[i,j]!=0):
            acumulador+=1
   return   acumulador


#Funcion que efectua la operacion de horizontal a un blob
def Horizontal(I, I1):
    vert, hor = I.shape
    for i in range(0, vert):
        bandera = 0
        for j in range(0, hor):
            if(I[i,j] != 0):
                if(bandera == 0):
                    y_inicial = j
                    y_final = j
                    bandera = 1
                else:
                    y_final = j
        if(bandera == 1):
            I1[i,y_inicial:y_final] = 255
  
  
#Funcion que efectua la operacion horizontal a un blob
def Vertical(I, I2):
    vert, hor = I.shape
    for i in range(0, hor):
        bandera = 0
        for j in range(0, vert):
            if(I[j,i] != 0):
                if(bandera == 0):
                    x_inicial = j
                    x_final = j
                    bandera = 1
                else:
                    x_final = j
        if(bandera == 1):
            I2[x_inicial:x_final,i] = 255
            
            
#Funcion que calcula los bordes de un frame
def Bordes(I):   
    img=cv2.Canny(I, threshold1 = 55, threshold2 = 155)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    return img


#Funcion que elimina los pequenos blobs de una imagen
def limpiar(Iin,Iout):
   objects_stats = [] #Para almacenar los stats de los objetos grandes
   areas = [] #Para almacenar unicamente areas de los objetos
   nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(Iin, 4, cv2.CV_32S)
   flag=0 
   for stat2 in stats:
     if stat2[4] < 100 and flag > 0: #Para no dibujar el rectangulo del background y eliminar el ruido
       objects_stats.append(stat2)
       areas.append(stat2[4])
       Iout[stat2[1]:stat2[1]+stat2[3], stat2[0]:stat2[0]+stat2[2]]=0
     flag = flag + 1 


#Devuelve el largo de la caja, si los dos objetos se unen
def LargoCaja(x1, width1, x2, width2): 
    if x1 <= x2:
        if x1 + width1 <= x2 + width2:
            return x1, x2 + width2 #(Largo de la caja)
        else:
            return x1, x1 + width1 #(Largo de la caja)
    else:
        if x2 + width2 <= x1 + width1:
            return x2, x1 + width1 #(Largo de la caja)
        else:
            return x2, x2 + width2 #(Largo de la caja)


#Devuelve el alto de la caja, si los dos objetos se unen
def AnchoCaja(y1, height1, y2, height2): 
    if y1 <= y2:
        if y1 + height1 <= y2 + height2:
            return y1, y2 + height2 #(Ancho de la caja)
        else:
            return y1, y1 + height1 #(Ancho de la caja)
    else:
        if y2 + height2 <= y1 + height1:
            return y2, y1 + height1 #(Ancho de la caja)
        else:
            return y2, y2 + height2 #(Ancho de la caja)


#Recibe los stats de objetos grandes obtenidos por CC y un minimo Td
def unionCajas(objects_stats, min_found): 
    new_objects_stats = []
    for i in range(len(objects_stats)-1):
        for j in range(i+1, len(objects_stats)):
            xI = objects_stats[i][0]
            yI = objects_stats[i][1]
            widthI = objects_stats[i][2]
            heightI = objects_stats[i][3]

            xJ = objects_stats[j][0]
            yJ = objects_stats[j][1]
            widthJ = objects_stats[j][2]
            heightJ = objects_stats[j][3]

            if xI <= xJ and xJ <= xI + widthI + min_found: #Si el punto inicial de un objeto se encuentra dentro del largo del otro o a la derecha
                if yI <= yJ and yJ <= yI + heightI + min_found: #Si el objeto esta abajo...
                    if abs(yJ - (yI + heightI)) <= min_found:
                        #Aplicamos union de objetos
                        x, width = LargoCaja(xI, widthI, xJ, widthJ)
                        y, height =  AnchoCaja(yI, heightI, yJ, heightJ)
                        area = objects_stats[i][4] + objects_stats[j][4]
                        new_objects_stats.append(np.array([x, y, width, height, area], dtype=np.int32))
                elif yJ < yI and yI <= yJ + heightJ + min_found: #Si el objeto esta arriba...
                    if abs(yI - (yJ + heightJ)) <= min_found:
                        #Union de objetos
                        x, width = LargoCaja(xI, widthI, xJ, widthJ)
                        y, height = AnchoCaja(yI, heightI, yJ, heightJ)
                        area = objects_stats[i][4] + objects_stats[j][4]
                        new_objects_stats.append(np.array([x, y, width, height, area], dtype=np.int32))
            elif xI > xJ and xI <= xJ + widthJ + min_found: #Si se encuentra a la izquierda...
                if yI <= yJ and yJ <= yI + heightI + min_found: 
                    if abs(yJ - (yI + heightI)) <= min_found:
                        #Aplicamos union de objetos
                        x, width = LargoCaja(xI, widthI, xJ, widthJ)
                        y, height =  AnchoCaja(yI, heightI, yJ, heightJ)
                        area = objects_stats[i][4] + objects_stats[j][4]
                        new_objects_stats.append(np.array([x, y, width, height, area], dtype=np.int32))
                elif yJ < yI and yI <= yJ + heightJ + min_found:
                    if abs(yI - (yJ + heightJ)) <= min_found:
                        #Union de objetos
                        x, width = LargoCaja(xI, widthI, xJ, widthJ)
                        y, height =  AnchoCaja(yI, heightI, yJ, heightJ)
                        area = objects_stats[i][4] + objects_stats[j][4]
                        new_objects_stats.append(np.array([x, y, width, height, area], dtype=np.int32))

    return new_objects_stats
    
    
############################################################################################################333

#cap = cv2.VideoCapture('highwayII_raw.avi')
cap = cv2.VideoCapture('highwayI_raw.avi')
#cap = cv2.VideoCapture('highway.avi')
#cap=cv2.VideoCapture('geneseeclip_short.avi')
#cap=cv2.VideoCapture('campus_raw.avi')
#cap = cv2.VideoCapture('mall.mpg')
#cap = cv2.VideoCapture('pedestrians_290to690.avi')

#Utilizamos el algoritmo KNN para substraccion de fondo
fgbg = cv2.createBackgroundSubtractorKNN()

#Declaramos algunos elementos estructurales
B1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #MORPH_ELLIPSE
B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel = np.ones((3,3), np.float32)


#Declaramos variables para la evaluacion del algoritmo
cont=0
TP=0.0
TN=0.0
FP=0.0
FN=0.0
div=0.0
PDR=0.0
NDR=0.0
F=0.0
P=0.0

#Abrimos las imagenes GT 
h45=cv2.imread('h45.png',0)
h65=cv2.imread('h65.png',0)
h85=cv2.imread('h85.png',0)
h105=cv2.imread('h105.png',0)
h125=cv2.imread('h125.png',0)
h145=cv2.imread('h145.png',0)
h165=cv2.imread('h165.png',0)
h185=cv2.imread('h185.png',0)

#Le quitamos los objetos al GT y nos quedamos con la sombra
sombras(h45)
sombras(h65)
sombras(h85)
sombras(h105)
sombras(h125)
sombras(h145)
sombras(h165)
sombras(h185)

#Mostramos las sombras del GT
cv2.imwrite('45.jpg',h45)
cv2.imwrite('65.jpg',h65)
cv2.imwrite('85.jpg',h85)
cv2.imwrite('105.jpg',h105)
cv2.imwrite('125.jpg',h125)
cv2.imwrite('145.jpg',h145)
cv2.imwrite('165.jpg',h165)
cv2.imwrite('185.jpg',h185)

#Bucle para recorrer todo el video
while(1):
    ret, frame = cap.read() #Leemos un frame del video

    if not ret:   #Si no hay mas frames salimos del video
        break
    
    a,b,c=frame.shape #Obtenemos las dimensiones del frame

    #Creamos las matrices en que vamos a almacenar nuestras imagenes a calcular
    fra2=np.zeros((a, b,c), np.uint8)
    I1=np.zeros((a, b), np.uint8)
    I2=np.zeros((a, b), np.uint8)
    IFEt=np.zeros((a,b), np.uint8)
    FRt=np.zeros((a,b), np.uint8)
    IFt=np.zeros((a,b), np.uint8)
    ISt=np.zeros((a, b), np.uint8)
    HRt=np.zeros((a,b), np.uint8)
    HRt=np.zeros((a,b), np.uint8)
    VRt=np.zeros((a,b), np.uint8)
    REt=np.zeros((a,b), np.uint8)
    HREt=np.zeros((a,b), np.uint8)
    VREt=np.zeros((a,b), np.uint8)
    HFRt=np.zeros((a,b), np.uint8)
    VFRt=np.zeros((a,b), np.uint8)
    HREMt=np.zeros((a,b), np.uint8)
    VREMt=np.zeros((a,b), np.uint8)
    
    #Hacemos una copia del frame actual
    fra2[:,:,:]=frame[:,:,:]

    #Detectamos el objeto 
    fgmask = fgbg.apply(frame)
  
    #Eliminamos los pequenos blobs (ruido)
    limpiar(fgmask,fgmask)
 
    
    Ct = cv2.medianBlur(fgmask,7)
    Ct = cv2.blur(Ct,(3,3))
    th, Ct = cv2.threshold(Ct,30, 255, cv2.THRESH_BINARY)
    limpiar(Ct,Ct) #Eliminamos los pequenos blobs (ruido) 
    
    #Calculamos ECt
    aux = cv2.erode(Ct, kernel, iterations = 1) 
    ECt = Ct - aux
    th, ECt = cv2.threshold(ECt, 180, 255, cv2.THRESH_BINARY)#Umbralizamos IEt
    #Calculamos DBt
    DEt = cv2.dilate(ECt, kernel, iterations = 1) 
    DBt = cv2.bitwise_and(DEt, Ct)
    
    #Calculamos Et e IEt
    
    Et = Bordes(frame)
    IEt = Et - DBt
    th, IEt = cv2.threshold(IEt, 10, 255, cv2.THRESH_BINARY)#Umbralizamos IEt
    
    connectivity = 8
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ct, connectivity, cv2.CV_32S) #Calculamos componentes conectados a Ct (mascara del objeto)
    objects_stats = []
    areas = []
    flag = 0          
    #Recorremos todos los blobs
    for stat in stats:
        if stat[4] > 105 and flag > 0: #Los blobs que tengan mas de 155 pixeles
            objects_stats.append(stat)
            areas.append(stat[4])
            #Aplicamos operacion horizontal a IEt para generar HRt
            Horizontal(IEt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]], HRt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]]) 
            #Aplicamos operacion vertical a IEt para generar VRt
            Vertical(IEt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]], VRt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]])
            #Dibujamos la caja envolvente de los blobs sobre el frame
            cv2.rectangle(frame,(stat[0],stat[1]),(stat[0]+stat[2],stat[1]+stat[3]),(0,255,0),1)   
        flag = flag + 1
        
    #Se eliminan los blobs pequenos
    limpiar(HRt,HRt)
    limpiar(VRt,VRt)
    REt = cv2.bitwise_and(HRt, VRt) #Hacemos una OR entre HRt y VRt para calcular REt
  
    
    #Calculamos componentes conectados a Ct
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ct, connectivity, cv2.CV_32S)
    objects_stats = []
    areas = []
    flag = 0          
    
    for stat in stats:
        if stat[4] > 105 and flag > 0:
            objects_stats.append(stat)
            areas.append(stat[4])
            #Aplicamos la operacion Horizontal a REt para generar HREt
            Horizontal(REt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]], HREt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]])
            #Aplicamos la operacion Vertical a REt para generar VREt
            Vertical(REt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]], VREt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]]) 
        flag = flag + 1
    
    #Calculamos cuales cajas se pueden unir 
    resuk = unionCajas(objects_stats, 0)
    
    for stat in resuk:
        cv2.rectangle(frame,(stat[0],stat[1]),(stat[2],stat[3]),(255,0,0),1) #Dibujamos a las cajas que se unen con el color azul
        #Aplicamos la operacion horizontal a HREt para generar HREMt
        Horizontal(HREt[stat[1]:stat[3], stat[0]:stat[2]], HREMt[stat[1]:stat[3], stat[0]:stat[2]])
        #Aplicamos la operacion horizontal a VREt para generar VREMt
        Vertical(VREt[stat[1]:stat[3], stat[0]:stat[2]], VREMt[stat[1]:stat[3], stat[0]:stat[2]])
     
    #Aplicamos una OR para unir los blobs   
    HREMt = cv2.bitwise_or(HREt, HREMt)
    VREMt = cv2.bitwise_or(VREt, VREMt)
    
    #Calculamos IFEt
    IFEt=cv2.bitwise_and(HREMt,VREMt)
   
    #Calculamos componentes conectados a IFEt
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(IFEt, connectivity, cv2.CV_32S)
    objects_stats = []
    areas = []
    flag = 0          
    
    for stat in stats:
        if stat[4] > 100 and flag > 0: #Si el blob es mayor de 100 pixeles
            objects_stats.append(stat)
            areas.append(stat[4])
            #Aplicamos la operacion horizontal a IFEt para generar HFRt
            Horizontal(IFEt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]], HFRt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]])
            #Aplicamos la operacion horizontal a IFEt para generar VFRt
            Vertical(IFEt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]], VFRt[stat[1]:stat[1]+stat[3], stat[0]:stat[0]+stat[2]])   
        flag = flag + 1
    
    #Calculamos FRt haciendo una OR entre HFRt y VFRt
    FRt=cv2.bitwise_or(HFRt,VFRt)
    
    #Calculamos ISt 
    ISt=Ct - DBt-FRt
    
    #Aplicamos un umbral a ISt
    th, ISt = cv2.threshold(ISt, 50, 255, cv2.THRESH_BINARY)
    
    #Eliminamos los pequenos blobs
    limpiar(ISt,ISt)
    
    #Hacemos la evaluacion del algoritmo para los 8 GT 
    cont+=1
   
    if(cont==45):
      TP+=TruePositives(ISt,h45)
      TN+=TrueNegatives(ISt,h45)
      FP+=FalsePositives(ISt,h45)
      FN+=FalseNegatives(ISt,h45)
#      cv2.imshow('ISt45',ISt)

    elif(cont==65):
      TP+=TruePositives(ISt,h65)
      TN+=TrueNegatives(ISt,h65)
      FP+=FalsePositives(ISt,h65)
      FN+=FalseNegatives(ISt,h65)
#      cv2.imshow('ISt65',ISt)

    elif(cont==85):
      TP+=TruePositives(ISt,h85)
      TN+=TrueNegatives(ISt,h85)
      FP+=FalsePositives(ISt,h85)
      FN+=FalseNegatives(ISt,h85)
#      cv2.imshow('ISt85',ISt)

    elif(cont==105):
      TP+=TruePositives(ISt,h105)
      TN+=TrueNegatives(ISt,h105)
      FP+=FalsePositives(ISt,h105)
      FN+=FalseNegatives(ISt,h105)
#      cv2.imshow('ISt105',ISt)

    elif(cont==125):
      TP+=TruePositives(ISt,h125)
      TN+=TrueNegatives(ISt,h125)
      FP+=FalsePositives(ISt,h125)
      FN+=FalseNegatives(ISt,h125)
#      cv2.imshow('ISt125',ISt)

    elif(cont==145):
      TP+=TruePositives(ISt,h145)
      TN+=TrueNegatives(ISt,h145)
      FP+=FalsePositives(ISt,h145)
      FN+=FalseNegatives(ISt,h145)
#      cv2.imshow('ISt145',ISt)

    elif(cont==165):
      TP+=TruePositives(ISt,h165)
      TN+=TrueNegatives(ISt,h165)
      FP+=FalsePositives(ISt,h165)
      FN+=FalseNegatives(ISt,h165)
#      cv2.imshow('ISt165',ISt)

    elif(cont==185):
      TP+=TruePositives(ISt,h185)
      TN+=TrueNegatives(ISt,h185)
      FP+=FalsePositives(ISt,h185)
      FN+=FalseNegatives(ISt,h185)
#      cv2.imshow('ISt185',ISt)
    #Convertimos ISt a color
    ISt=cv2.cvtColor(ISt,cv2.COLOR_GRAY2BGR)
    
    #Le aplicamos las sombras calculadas al frame
    Imagen_final=cv2.bitwise_or(ISt,fra2)
   
    #Mostramos las imagenes del algoritmo   
#    cv2.imshow('Mascara', fgmask)
    #cv2.imshow('Mascara_suavizada', fgmaskf)
    #cv2.imshow('VideoOriginal_Grises', gray_frame)
    cv2.imshow('Video Original', frame)
#    cv2.imshow('Ct', Ct)
#    cv2.imshow('ECt',ECt)
#    cv2.imshow('Imagen suavizada', gray_framef)
#    cv2.imshow('Imagen suavizada', frame)
#    cv2.imshow('Et', Et)
#    cv2.imshow('DBt', DBt)
#    cv2.imshow('DEt', DEt)
#    cv2.imshow('IEt',IEt)
#    cv2.imshow('VRt',VRt)
#    cv2.imshow('HRt',HRt)
#    cv2.imshow('REt',REt)
#    cv2.imshow('VREt',VREt)
#    cv2.imshow('HREt',HREt)
#    cv2.imshow('VREMt',VREMt)
#    cv2.imshow('HREMt',HREMt)
#    cv2.imshow('IFEt',IFEt)
#    cv2.imshow('VFRt',VFRt)
#    cv2.imshow('HFRt',HFRt)
#    cv2.imshow('FRt',FRt)
    cv2.imshow('Imagen_final',Imagen_final)
#    cv2.imshow('frame',fra2)
    
#    cv2.imshow('ConnectedComponents', prueba2)
    
    k = cv2.waitKey(100)
    if k == 48 or k == 113: # '0' o  'q' para cerrar el video
        break

#Calculamos las curvas ROC
TP/=8 #True Positives : Pixeles bien clasificados como FG
TN/=8 #True Negatives : Pixeles bien clasificados como BG
FP/=8 #False Positives : Pixeles mal clasificados como FG
FN/=8 #False Negatives : Pixeles mal clasificados como BG

PDR=TP/(TP+FN) #Positive Detection Rate
NDR=TN/(TN+FP) #Negative Detection Rate
P=TP/(TP+FP)   #Precision
F=2*PDR*P/(PDR+P) #Medida de efectividad
#Imprimimos los resultados
print("Efectividad: ",F,"\nTasa de deteccion positiva: ",PDR,"\nTasa de deteccion negativa: ",NDR,"\nPrecision: ",P)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
