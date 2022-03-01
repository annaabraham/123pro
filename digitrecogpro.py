import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import image
import PIL.ImageOps
import os,xxl,time

if(not os.environ.get('PYTHONHTTPSVERIFY','')and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context=ssl._create_unverified_context

X,Y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(Y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,random_state=9,train_size=7500,test_size=2500)
xtrainscaled=xtrain/255.0
xtestscaled=xtest/255.0
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscaled,ytrain)

ypredict=clf.predict(xtestscaled)
accuracy=accuracy_score(ytest,ypredict)
print(accuracy)

tap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.color_BGR2GREY)

        height,width=gray.shape
        upperleft=(int(width/2-56),int(height/2-56))
        bottomright=(int(width/2+56),int(height/2+56))

        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)

        roi=gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        
        impil=Image.fromarray(roi)
        imagebw=impil.convert('L')
        imagebwresize=imagebw.resize((28,28),Image.ANTIALIAS)

        imagebwresizeinverted=PIL.ImageOps.invert(imagebwresize)
        pixelfilter=20
        minpixel=np.percentile(imagebwresizeinverted,pixelfilter)
        imageScaled=np.clip(imagebwresizeinverted-minpixel,0,255)
        maxpixel=np.max(imagebwresizeinverted)