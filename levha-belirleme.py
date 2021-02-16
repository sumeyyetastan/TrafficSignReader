import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3
import os.path
import speech_recognition as sr

from scipy.stats import itemfreq
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.metrics import structural_similarity as ssim
engine = pyttsx3.init()

#Function to compare pictures
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err
#Function to find image similarity
def compare_images(imageA, imageB, title):
    a = mse(imageA, imageB)
    #find image similarity
    b = ssim(imageA, imageB)
    #It displays the original picture and the picture contained in the data on the screen.
    fig = plt.figure(title)
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()

    #The function returns the image similarity ratio.
    return b

kernel = np.ones((5,5),np.uint8) # An array is specified for the kernel.
kernel1 = np.array([[1, 1, 1, 1, 1], #same array is determined like this
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]])

kernel1 = kernel1/sum(kernel1)

#Work done for voiceover
engine = pyttsx3.init()
hiz = engine.getProperty('rate')   # getting details of current speaking rate
print (hiz)                        #printing current voice rate
engine.setProperty('rate', 115)
engine.say("Please tell the number of the traffic sign you want to know the meaning of.")

engine.runAndWait()
# Record Audio
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# Speech recognition using Google Speech Recognition
try:

    print("You said: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

#The work done to find the path to the picture from the folder
klasor="Resimler/"
numara=r.recognize_google(audio)
uzanti=".jpg"
yol=klasor+numara+uzanti
img = cv2.imread(yol)
cv2.imshow("Original Image",img)
cv2.waitKey()

#for the mask of the red-scala color
boundaries = [ ([0, 0, 0] ,[120, 70, 250]) ] # define the list of boundaries

for (lower, upper) in boundaries:
    # create Numpy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    # find the colors within the specified boundaries and apply
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)  # the mask

    cv2.imshow("images", output)
    cv2.waitKey(0)
#The method we use to reduce noise
output =cv2.fastNlMeansDenoising(output,None,30.0, 7, 21)
cv2.imshow("Noise Reduced Image",output)
cv2.waitKey(0)

#Start of the function that converts the picture to black and white in order to detect the traffic sign
en,boy,katman = np.shape(output) #Take an image size
yeniResim = np.ones((en,boy,katman)) #we create a white visual.
for i in range(en):
    for j in range(boy):
        if(output[i,j,2] >70): #We compare the second layer of the picture whether it is greater or less than the threshold value of 70.
            yeniResim[i,j] = 0 # If it is less than 70, paint the louse black

        else:
            yeniResim[i,j]= 1 # If it is less than 70, paint the louse wihte

im_floodfill_inv=yeniResim
cv2.imshow("Black and White Image",im_floodfill_inv)
cv2.waitKey(0)

#The operations we do to determine the data type of the image
J = im_floodfill_inv*255

print(im_floodfill_inv.dtype)
mediandilate = J.astype(np.uint8)
print(mediandilate.dtype)

imgshape=mediandilate
img_contour = imgshape.copy()
#we use canny edge method to define edges
img_canny = cv2.Canny(imgshape, 200, 400)
cv2.imshow("Image-Canny" , img_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5,5),np.uint8)
img_dilated =img_canny

#The function we use to determine the geometric shape of the object in the picture
def get_contours(imgshape, img_contour):
  contours, hierarchy = cv2.findContours(imgshape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 9000:
      cv2.drawContours(img_contour, cnt, -1, (255, 0, 255), 1)

      # Find length of contours
      param = cv2.arcLength(cnt, True)

      # Approximate what type of shape this is
      approx = cv2.approxPolyDP(cnt, 0.01 * param, True)
      shape, x, y, w, h = find_shape(approx)
      cv2.putText(img_contour, shape, (x+78, y+200), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 0, 255), 1)

  return approx, param, img_contour, contours, cnt

#The function that prints the number of sides of the figure in the picture
def find_shape(approx):
  x, y, w, h = cv2.boundingRect(approx)
  print("Number of edge ",len(approx))

  #If the number of sides is between 3 and 6, it's a triangle.
  if  (len(approx) >= 3 ) and ( len(approx) <= 6 ):
    print('Triangle')
    s = "Triangle" #determine the object is triangle
    orig_image = mediandilate.copy()
    #Turning the picture gray, we give the threshold value.
    gray=cv2.cvtColor(mediandilate,cv2.COLOR_BGR2GRAY)
    ret, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.waitKey(0)
    #calculate accuracy as a percent of contour perimeter
    #Since the picture is a triangle, we determine the sides by multiplying by 0.03.
    accuracy=0.03*cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(mediandilate,[approx],0,(255,0,0),10)
    cv2.imshow('Detection Shape', mediandilate)
    cv2.waitKey(0)
    #We mask the picture on orijinal image
    mask = np.zeros(mediandilate.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(approx) # returns (x,y,w,h) of the rect
    #We cut the masked part from the original picture.
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    dimensions = cropped.shape
    height = cropped.shape[0]
    width = cropped.shape[1]
    channels = cropped.shape[2]

    #We made it to be able to see the size, height, width and depth of the picture.
    print('Image Dimension    : ',dimensions)
    print('Image Height       : ',height)
    print('Image Width        : ',width)
    print('Number of Channels : ',channels)

    #We resized our picture to 200x200.
    dsize = (200, 200)
    cropped = cv2.resize(cropped, dsize)

    dimensions = cropped.shape
    height = cropped.shape[0]
    width = cropped.shape[1]
    channels = cropped.shape[2]

    print('Image Dimension New    : ',dimensions)
    print('Image Height  New     : ',height)
    print('Image Width   New     : ',width)
    print('Number of Channels  New: ',channels)


    # create the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    dst = wbg+res
    detectOrgImg=img.copy()

    cv2.imshow("Mask",mask)
    cv2.imshow("Cropped", cropped )
    cv2.imshow("Samed Size Black Image", res)
    cv2.imshow("Samed Size White Image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    data_num=1
    temp=0
    tempname=" "
    #The array that holds the names of the data
    nameoftraficsign=["Attention Crosswalk ","Attention Road Works ","Attention uneven road","Attention No Left Turn ",
    "Attention left hand curve ","Attention speed limit 70","Attention no u-turn ","Attention steep hill downwards"]
    #The function we use to get data images  in the folder
    for data_num in range(1,9):
        uzanti=".jpg"
        data=str(data_num)+uzanti
        data='data/'+data
        data=cv2.imread(data)
        dsize = (200, 200)
        data = cv2.resize(data, dsize) #The picture has been resized.
        original=cropped
        images = ("Original", original), ("Other", data)

        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        #Similarity Ratio function
        benzerlikorani = compare_images(original, data, "Original vs. Data")
        print("Similarity Ratio", benzerlikorani)

        #Function to find which images in the data are most similar
        if benzerlikorani > temp:
            temp=benzerlikorani
            tempname=(data_num)-(1) #Bringing the meaning of the picture in the Array

    print("The most similar value",temp)
    print("Meaning of the most similar data picture :",tempname)
    print("Meaning of the traffic sign ",nameoftraficsign[tempname])

    #Speech Recognition voice rate adjustment
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 115)
    #Speaks the meaning of the traffic sign
    engine.say(nameoftraficsign[tempname])
    engine.runAndWait()
    #Writes the meaning of the traffic sign on the original picture
    cv2.putText(detectOrgImg, nameoftraficsign[tempname],(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow("Result",detectOrgImg)
    cv2.waitKey()

  elif len(approx) == 8:
    s = "Octagon"

  else:
    s = "Circle"
    orig_image = mediandilate.copy()
    #Turning the picture gray, we give the threshold value.
    gray=cv2.cvtColor(mediandilate,cv2.COLOR_BGR2GRAY)
    ret, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.waitKey(0)
    #Because it is circle, we make the edge detection rate more precise, in it we multiply it by 0.01
    accuracy=0.001*cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(mediandilate,[approx],0,(255,0,0),10)
    cv2.imshow('Detection Shape', mediandilate)
    cv2.waitKey(0)

    mask = np.zeros(mediandilate.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(approx) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    dimensions = cropped.shape
    height = cropped.shape[0]
    width = cropped.shape[1]
    channels = cropped.shape[2]

    print('Image Dimension    : ',dimensions)
    print('Image Height       : ',height)
    print('Image Width        : ',width)
    print('Number of Channels : ',channels)

    dsize = (200, 200)
    cropped = cv2.resize(cropped, dsize)

    dimensions = cropped.shape
    height = cropped.shape[0]
    width = cropped.shape[1]
    channels = cropped.shape[2]

    print('Image Dimension New    : ',dimensions)
    print('Image Height  New     : ',height)
    print('Image Width   New     : ',width)
    print('Number of Channels  New: ',channels)

    wbg = np.ones_like(img, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    dst = wbg+res
    detectOrgImg=img.copy()

    cv2.imshow("Mask",mask)
    cv2.imshow("Cropped", cropped )
    cv2.imshow("Samed Size Black Image", res)
    cv2.imshow("Samed Size White Image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    data_num=1
    temp=0
    tempname=" "
    nameoftraficsign=["Attention Crosswalk ","Attention Road Works ","Attention uneven road","Attention No Left Turn ","Attention left hand curve ",
    "Attention speed limit 70","Attention no u-turn ","Attention steep hill downwards"]
    for data_num in range(1,9):
        uzanti=".jpg"
        data=str(data_num)+uzanti
        data='data/'+data
        data=cv2.imread(data)
        dsize = (200, 200)
        data = cv2.resize(data, dsize)
        original=cropped
        images = ("Original", original), ("Other", data)

        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        benzerlikorani = compare_images(original, data, "Original vs. Data")
        print("benzerlik oranı", benzerlikorani)

        if benzerlikorani > temp:
            temp=benzerlikorani
            tempname=(data_num)-(1)

    print("The most similar value",temp)
    print("Meaning of the most similar data picture :",tempname)
    print("Meaning of the traffic sign ",nameoftraficsign[tempname])

    rate = engine.getProperty('rate')
    engine.setProperty('rate', 115)
    engine.say(nameoftraficsign[tempname])
    engine.runAndWait()
    cv2.putText(detectOrgImg, nameoftraficsign[tempname],(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.imshow("Result",detectOrgImg)
    cv2.waitKey()

get_contours(img_dilated, img_contour)
cv2.imshow("Resim-contour" , img_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
