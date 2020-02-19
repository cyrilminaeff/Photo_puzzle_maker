# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:41:44 2019

@author: cyril
"""

import cv2
import numpy as np
import time
import os
import random

start_time = time.time()
seglist =   os.listdir('.\\segments 200')

carpet = cv2.imread('IMG_3839-2.jpg')

# кроп и ресайз ковра
hc, wc = carpet.shape[:2]
new_carpet_width = 2000 # задание ширины ковра
hc, wc = int(hc/wc*new_carpet_width), new_carpet_width 
carpet = cv2.resize(carpet, (wc, hc) )
n_hor = 40 # количество сегментов в ковре по ширине
n_vert = int(hc/wc*n_hor)


# цикл по вставке каждого сегмента в ковер
for ni in range(n_hor):
    print(ni/n_hor*100,'% ', end = '') # progress
    for nj in range(n_vert):
                
        # чтение сегмента
        #ns =  ni*n_vert + (nj+1) 
        #segment = cv2.imread('.\\segments 200\\'+seglist[ns % len(seglist)]) # выбор сегментов по порядку
        segment = cv2.imread('.\\segments 200\\'+seglist[random.randint(0,len(seglist)-1)]) # выбор в случайном порядке

        # кроп и ресайз сегмента
        hs, ws = segment.shape[:2]
        if ws > hs:
            segment = segment[ : , int((ws-hs)/2) : int((ws+hs)/2)]
        else:
            segment = segment[ int((hs-ws)/2) : int((hs+ws)/2) , :]
        hs, ws = wc//n_hor, wc//n_hor
        segment = cv2.resize(segment, (ws,hs))
        
        # узнаем цветовую яркость сегмента
        gamma_s = np.array([0,0,0]) # средняя цветовая яркость сегмента (bgr)
        for k in range(3):
            gamma_s[k] = np.sum(segment[:,:,k])
        gamma_s = gamma_s / ws / hs # нормализация     
        #print('средняя гамма сегмента = ', gamma_s)        
        
        
        
        # узнаем гамму кропа
        crop = carpet[ hs*nj : hs*(nj+1) , ws*ni : ws*(ni+1)]
        gamma_c = np.array([0,0,0]) # средняя цветовая яркость кропа (bgr)
        for k in range(3):
            gamma_c[k] = np.sum(crop[:,:,k])
        gamma_c = gamma_c / ws / hs # нормализация        
        
        # меняем яркость сегмента
        value = 1 # степень перекраски пикселей
        b, g, r = cv2.split(segment)
        factor = ((gamma_c - gamma_s)*value) # на сколько нужно поменять яркость всех пикселей
        b, g, r = cv2.add(b,factor[0]), cv2.add(g,factor[1]), cv2.add(r,factor[2])
        segment_colored = cv2.merge( (b,g,r) )
        
        #вставка сегмента в ковер
        carpet[ hs*nj : hs*(nj+1) , ws*ni : ws*(ni+1)] = segment_colored

del segment_colored, segment, crop, b, g, r
print('---%s---' % (time.time()-start_time))

cv2.imshow('image_puzzle', carpet)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('puzzle.jpg',carpet)
    cv2.destroyAllWindows()
