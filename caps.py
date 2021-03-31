from cdetect import V5
import cv2
import os
import cv2
import torch
import glob
import os
from PIL import Image  
import json
import sqlite3
import base64
from datetime import date
import PIL
import time
t1 = time.time()

from flask import Flask,request,jsonify

app = Flask(__name__)


def load_model():
    global PATH_SAVED_MODEL, yoloTiny, IMAGE_SIZE
    # PATH to saved and exported tensorflow model
    PATH_SAVED_MODEL = os.path.join(os.getcwd(), 'best.pt')
    IMAGE_SIZE = 640
    yoloTiny = V5( PATH_SAVED_MODEL)

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData

def insertBLOB(Id, photo):
    try:
        sqliteConnection = sqlite3.connect('caps.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        #sqliteConnection.execute("""CREATE TABLE IF NOT EXISTS cones(id INTEGER PRIMARY KEY, roi BINARY UNIQUE)""")
        cursor.execute("""SELECT MAX([id]) FROM [Caps]""")
        sqlite_insert_blob_query = """ INSERT INTO Caps
                                  (id, roi) VALUES (?, ?)"""

        roi = convertToBinaryData(photo)
        # Convert data into tuple format
        data_tuple = (Id, roi)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")

def coords(list):
    li2 = ['class','x1','y1','x2','y2','x3','y3','x4','y4']
    x3 = list[1] 
    list.append(x3)
    y3 = list[4]
    list.append(y3)
    x4 = list[3]
    list.append(x4)
    y4 = list[2]
    list.append(y4)
    #print(list)
    thresh = list[5]
    del list[5]
    list.append(thresh)
    #print(list)
    dictionary = dict(zip(li2, list))
    return dictionary, x3, y4, x4, y3 #coordinates are same in bbox, foe eg: x3=x1

def saveimg(image_roi):
    cv2.imwrite(f"img.jpg", image_roi)

# def roi(name_img, name_file):
#     path = "runs/detect/exp/images/"
#     for img in glob.glob(path + name_img):
#         # with open(path + "labels/" + name_file, 'r') as file:
#         with open(f"runs/detect/exp/labels/{image_name}" + '.txt', 'r') as file:
#             data = file.read().replace("\n",'')
#             #print(data)
#             li = data.split(" ")
#             #print(li)
#             final_dictionary, xmin, ymin, xmax , ymax = coords(li)
#             xmn = int(xmin)
#             ymn = int(ymin)
#             xmx = int(xmax)
#             ymx = int(ymax)
#             img = cv2.imread(img)
#             # cv2.imshow("Full Image",img)
#             cv2.waitKey(0)
#             print(f"cropped image details: {final_dictionary}\n")
#             cropped_image = img[ymn:ymx , xmn:xmx]
#             #cv2.imshow("ROI", cropped_image)
#             cv2.imwrite(f"runs/detect/exp/cropped/{name_img}", cropped_image)
#             return img, cropped_image
#             #cv2.waitKey(0)


load_model()

""" create a database connection to a SQLite database """
conn = None
try:
    conn = sqlite3.connect(r"caps.db")
    print(sqlite3.version)
except Error as e:
    print(e)
finally:   
#create_connection(r"threads.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS Caps(id INTEGER AUTO_INCREMENT, photo BINARY )""")
    if conn:
        conn.close()

image_name = 0

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        global image_name
        img = cv2.imread('valid\images\WhatsApp-Image-2021-03-17-at-7-03-34-PM_jpeg.rf.61028063e6fc35349f0aae306c33b0d0.jpg', 0)
        a= yoloTiny.detect(img,image_name)
        cv2.imwrite(f"runs/detect/exp3/{image_name}.jpg", a)
        #full_image, cropped_image = roi(f"{image_name}.jpg","exp.txt")
        insertBLOB(image_name,f"runs/detect/exp3/{image_name}.jpg")
        print(image_name)
        # with open(f"runs/detect/exp/cropped/{image_name}.jpg","rb") as img_file:
            #my_string = base64.b64encode(img_file.read())
        with open(f"runs/detect/exp3/{image_name}.jpg","rb") as img1_file:
            my_string1 = base64.b64encode(img1_file.read())
        image_name +=1
        return json.dumps({"image": my_string1.decode('utf-8')}) #, "image1": my_string1.decode('utf-8')}) 
        

