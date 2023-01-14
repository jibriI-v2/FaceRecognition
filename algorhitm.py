#!/usr/bin/env python
from PIL import ImageChops, Image, ImageOps #Importar bibliotecas
import math, operator
import functools
import cv2
import os
from collections import defaultdict
import re
import numpy as np

def rmsdiff(im1, im2):                      # Función de diferencia entre imágenes
    h = ImageChops.difference(im1, im2).histogram()
    return math.sqrt(functools.reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))
def files():                                # Detecta archivos añadidos a la carpeta de imágenes
    global savedSet, databpath, imgpath
    dataSet=set()
    for file in os.listdir(databpath):
        fullpath=os.path.join(databpath,file)
        if os.path.isfile(fullpath):
            dataSet.add(os.path.join(databpath,file))
    retrievedSet=set()
    for file in os.listdir(imgpath):
        fullpath=os.path.join(imgpath, file)
        if os.path.isfile(fullpath):
            retrievedSet.add(os.path.join(imgpath,file))
    newSet=retrievedSet-savedSet 
    savedSet=retrievedSet
    return dataSet, newSet
def main():
    global umbral
    base,analizar= files()
    databaseplaceholder={}
    databasematchsource={}
    imagesource={}
    noreconocidas={}
    diferenciaconminimo={}
    for imag in analizar:
        diferenciaconminimo[imag]=1000
        im1=ImageOps.grayscale(Image.open(imag))
        for dat in base:                            # Consigue la diferencia con todas las entradas
            im2=ImageOps.grayscale(Image.open(dat))
            if rmsdiff(im1,im2)<diferenciaconminimo[imag]: 
                diferenciaconminimo[imag]=rmsdiff(im1,im2)
                databaseplaceholder[imag]=dat
        if diferenciaconminimo[imag]<=umbral:
            databasematchsource[imag]=databaseplaceholder[imag]
            imagesource[imag]=imag
        else:
            noreconocidas[imag]=imag
    for reconocida in databasematchsource:                         # Salida para coincidencias
        mensajeantes= "La imagen "+imagesource[reconocida]+" pertenece a "+databasematchsource[reconocida]
        mensajesinpath=mensajeantes.replace(databpath+"\\","").replace(imgpath+"\\","")
        mensajefinal=re.sub('.jpg'+'$','',mensajesinpath)
        print(mensajefinal+".")
        imagencomparada=cv2.resize(cv2.imread(imagesource[reconocida], cv2.IMREAD_ANYCOLOR),(400,566),interpolation=cv2.INTER_AREA)
        imagenrecon=cv2.resize(cv2.imread(databasematchsource[reconocida], cv2.IMREAD_ANYCOLOR),(400,566),interpolation=cv2.INTER_AREA)
        imagentotal=np.concatenate((imagencomparada,imagenrecon),axis=1)
        cv2.imshow("Izquierda: "+imagesource[reconocida]+"          Derecha: "+databasematchsource[reconocida].replace(databpath+"\\","").replace(".jpg","")+'           Diferencia: '+str(diferenciaconminimo[reconocida]),imagentotal)
        cv2.waitKey(0)
    for norec in noreconocidas:                                    # Salida para imágenes sin coincidencia
        print("La imagen "+norec.replace(imgpath+"\\","")+" no se reconoce.")

databpath= "database"
imgpath= "images"
umbral=35
savedSet = set()
while __name__=="__main__":
    main()
