
import sys

sys.path.append('./lib_original')

import os
import pyDBoW3 as bow
import numpy as np
import cv2
import xv_file


sequence = 'hl2_5'
path_in = os.path.join('./data', sequence, 'img')
fname_images = sorted(xv_file.scan_files(path_in))


orb = cv2.ORB_create()
feature_list = []

# extract features
for fname_image in fname_images:
    image = cv2.imread(fname_image, 0)
    cv2.imshow('Video', image)
    cv2.waitKey(1)
    kp, qd = orb.detectAndCompute(image, None)
    feature_list.append(qd)

voc = bow.Vocabulary()
voc.create(feature_list)
voc.save('./hl2_5_ORBvoc.bin', True)
del voc

voc = bow.Vocabulary()
voc.load('./hl2_5_ORBvoc.bin')
db = bow.Database()
db.setVocabulary(voc, True, 0)
del voc

for features in feature_list:
    db.add(features)

results = db.query(feature_list[10], 1, -1)
results = results[0]
print(f'Id: {results.Id}')
print(f'Score: {results.Score}')
print(f'nWords: {results.nWords}')
print(f'bhatScore: {results.bhatScore}')
print(f'chiScore: {results.chiScore}')
print(f'sumCommonVi: {results.sumCommonVi}')
print(f'sumCommonWi: {results.sumCommonWi}')
print(f'expectedChiScore: {results.expectedChiScore}')

#for i0, f0 in enumerate(feature_list):




    #b0 = voc.transform(f0)
    #for i1, f1 in enumerate(feature_list):
    #    b1 = voc.transform(f1)
    #    score = voc.score(b0, b1)
    #    print(f'score {i0},{i1}: {score}')
        
