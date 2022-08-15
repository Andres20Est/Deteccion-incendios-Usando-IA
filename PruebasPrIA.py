#%% Librerias
import pathlib
import numpy as np
import pandas as pd 
import cv2
import tensorflow as tf
import joblib

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import model_selection
from pywhatkit import sendwhatmsg_instantly as SendWssp
from sklearn.metrics import confusion_matrix, f1_score
#%% Link descarga base de datos
# https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images
#%% Importar Imagenes
#Path a la carpeta de las imagenes
fire_image_path = r"C:\Users\andre\Desktop\8vo\Inteligencia Artificial\Proyecto\DBs\Data/Train_Data/Fire"
non_fire_path = r"C:\Users\andre\Desktop\8vo\Inteligencia Artificial\Proyecto\DBs\Data/Train_Data/Non_Fire"

fire_image_path = pathlib.Path(fire_image_path)
non_fire_path = pathlib.Path(non_fire_path)

Train = {
    "Fire":list(fire_image_path.glob("*.jpg")),
    "NonFire":list(non_fire_path.glob("*.jpg"))
}

Y_Train = {
    "Fire":0,"NonFire":1
}

X, y = [], []
for label, images in Train.items():
    for image in images:
        img = cv2.imread(str(image)) # Reading the image
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (250, 250))
            X.append(img)
            y.append(Y_Train[label])
#%% Funciones del main
TamHist=[10,10]
def Matr2Vect(Hist2D): #Convierte una matriz de nxm a un vector de 1x(nxm)
    Vect1=[]
    Vect=np.zeros((1,np.size(Hist2D)))
    Tam=np.size(Hist2D,axis=0)
    for i in range(Tam):
        Vect1=np.concatenate((Vect1,Hist2D[i,:]),axis=0)
    for j in range(np.size(Hist2D)):
        Vect[0,j]=Vect1[j]
    return Vect

def Img2DHist(Img): # Convierte una imagen a su histograma en canales de tono y saturacion H y S
    hsv = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV)
    hist2 = cv2.calcHist( [hsv], [0, 1], None, TamHist, [0, 180, 0, 256] )
    Vector = Matr2Vect(hist2)
    return Vector

#%% Variables Globales
Datos=np.zeros((1,TamHist[0]*TamHist[1]))
#%% Main
#%% Obtencion Matriz de caracteristicas y Etiquetas categoricas
for img in range(len(X)):
    Vector=Img2DHist(X[img])
    Datos=np.concatenate((Datos,Vector),axis=0)
Datos=Datos[1:,:] #Se elimina el dato sobrante
y=np.array(y).reshape(1+img,1)
Training=np.concatenate((Datos,y),axis=1)

Train, Valid_ = model_selection.train_test_split(Training, test_size = int(0.2*len(Training)), train_size = int(0.8*len(Training)))
Valid, Test = model_selection.train_test_split(Valid_, test_size = int(0.5*len(Valid_)), train_size = int(0.5*len(Valid_)))

Y_Train=Train[:,TamHist[0]*TamHist[1]]
Y_Valid=Valid[:,TamHist[0]*TamHist[1]]
Y_Test=Test[:,TamHist[0]*TamHist[1]]

Y_Train_Dummies = pd.get_dummies(Y_Train)
Y_Valid_Dummies = pd.get_dummies(Y_Valid)
Y_Test_Dummies = pd.get_dummies(Y_Test)

Train=Train[:,:TamHist[0]*TamHist[1]]
Valid=Valid[:,:TamHist[0]*TamHist[1]]
Test=Test[:,:TamHist[0]*TamHist[1]]

Caract=np.size(Y_Train_Dummies,axis=1)
#%% Construccion de la Red neuronal

DNN = Sequential() # Se crea un modelo
DNN.add(Dense(TamHist[0]*TamHist[1], activation = 'tanh', input_shape = (TamHist[0]*TamHist[1],))) #Capa Entrada
DNN.add(Dense(250, activation = 'tanh'))                   #Capa Oculta 1
DNN.add(Dropout(0.20))
DNN.add(Dense(50, activation = 'tanh'))                    #Capa Oculta 2
DNN.add(Dense(Caract, activation = 'softmax'))                #Capa Salida

# Optimizacion
DNN.compile(optimizer = 'adam',
                  loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                  metrics = 'categorical_accuracy')
# Entrenamiento
DNN.fit(Train,Y_Train_Dummies, epochs = 250,  #250
              verbose = 1 , workers = 8 , use_multiprocessing=True,
              validation_data = (Valid,Y_Valid_Dummies))
# Prueba
y_hat = DNN.predict(Test)
y_out = y_hat.round()

# Transformación de datos: 
y_out = pd.DataFrame(y_out)
y_out = y_out.values.argmax(1)

# Métricas de desempeño: 
c_dnn = confusion_matrix(Y_Test, y_out)
F1_score_dnn = 100*f1_score(Y_Test, y_out, average = 'weighted')

#%% Extraccion Parametros

joblib.dump(DNN,'Red_Incendios.joblib')

#%% Alerta whatsapp
EnableAlerta=False
if F1_score_dnn >= 85:
    EnableAlerta=True
"""
if y_out!=1;
    EnableAlerta=True
"""
if EnableAlerta==True:
    Numero="+573505305548"
    Mensaje="\U0001f525 \u26A0\uFE0F Se ha detectado un posible incendio cerca a su zona \u26A0\uFE0F \U0001f525"
    SendWssp(Numero,Mensaje)

