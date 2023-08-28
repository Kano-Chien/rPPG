import cv2
import os
import pandas as pd
import numpy as np
import pickle




def video_to_image(vid_path,save_dir):
    timeF=1
    save_dir = os.path.join(save_dir,'frame')
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    video_path=os.path.join(vid_path,'vid.avi')
    cap = cv2.VideoCapture(video_path)
    c = 1
    ret = True
    while ret:  # 循环读取视频帧
        ret, frame = cap.read()
        if (c % timeF == 0) & ret == True:  # 每隔timeF帧进行存储操作
            save_path=os.path.join(save_dir,str(c-1)+'.jpg')
            cv2.imwrite(save_path,frame)  # 存储为图像
        c += 1
    cap.release()
    print("finish")

def image_to_gray(vid_path,save_dir):
    save_dir = os.path.join(save_dir, 'gray_png')
    video_dir = os.path.join(vid_path,'frame')
    os.makedirs(save_dir, exist_ok=True)
    pic_path = video_dir
    pic = os.listdir(pic_path)
    print(pic)
    for f in range(len(pic)):
        image_path = pic_path + '/'+str(f)+'.jpg'
        #print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_dir + '/'+str(f)+'.jpg', image)  # 存储为图像
def crop_image(gray_path,save_dir,path):
    gray_dir = os.path.join(gray_path,'frame')
    save_dir = os.path.join(save_dir, 'face')
    os.makedirs(save_dir, exist_ok=True)
    pic_path = gray_dir
    pic = os.listdir(pic_path)
    print(pic)
    errorlist = []
    for f in range(len(pic)):

        image_path = pic_path + '/'+str(f)+'.jpg'
        img = cv2.imread(image_path)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        facesize=[]
        i=0
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            facesize.append(w*h)
            num = facesize.index(max(facesize))
            i=i+1
        if len(faces):
            x, y, w, h=faces[num]
            crop_img = img[y:y + h, x:x + w]
        else:
            crop_img=img #若無法偵測人臉 輸出原圖
            errorlist.append(save_dir + '/' + str(f) + '.jpg') #記錄哪張圖無法輸出人臉
        print(save_dir + '/'+str(f)+'.jpg')
        cv2.imwrite(save_dir + '/'+str(f)+'.jpg', crop_img)
    print(errorlist)
    test = pd.DataFrame(columns=['name'], data=errorlist)
    test.to_csv(path +'/errorlist.csv',) #輸出無法偵測人臉的紀錄
def face_detection(image):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    facesize = []
    i = 0
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:

        facesize.append(w * h)
        num = facesize.index(max(facesize))
        i = i + 1
    if len(faces):
        x, y, w, h = faces[num]
        rect_image=cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        crop_img = image[y:y + h, x:x + w]
        no_face=False
    else:
        crop_img = image  # 若無法偵測人臉 輸出原圖
        rect_image=image
        no_face=True

    return crop_img,rect_image,no_face

def face_to_head_cheel_nosave(img):
    size = img.shape
    height = size[0]  # height(rows) of image
    width = size[1]  # width(colums) of image
    head_img = img[0:int(height / 3), 0:width]  # 額頭為最上面1/3
    cheek_img = img[int(height / 2):int(height * 5 / 6), 0:width]  # 臉頰為中間向下1/3
    return head_img,cheek_img
def face_to_head_cheek_save(face_path,save_dir):
    face_dir = os.path.join(face_path, 'face')
    savehead_dir = os.path.join(save_dir, 'head')
    os.makedirs(savehead_dir, exist_ok=True)
    savecheek_dir = os.path.join(save_dir, 'cheek')
    os.makedirs(savecheek_dir, exist_ok=True)
    pic_path = face_dir
    pic = os.listdir(pic_path)
    print(pic)
    for f in range(len(pic)):
        image_path = pic_path + '/'+str(f)+'.jpg'
        img = cv2.imread(image_path)
        size=img.shape
        height = size[0]  # height(rows) of image
        width = size[1]  # width(colums) of image
        head_img = img[0:int(height / 3), 0:width]  # 額頭為最上面1/3
        cheek_img = img[int(height / 2):int(height * 5 / 6), 0:width]  # 臉頰為中間向下1/3
        cv2.imwrite(savehead_dir + '/' + str(f) + '.jpg', head_img)
        cv2.imwrite(savecheek_dir + '/' + str(f) + '.jpg', cheek_img)
        print(savehead_dir + '/' + str(f) + '.jpg')
        print(savecheek_dir + '/' + str(f) + '.jpg')



def crop_all_face(dataset_dir):
    for i in all_file_path:
        print("start ", i)
        path=os.path.join(dataset_dir,i)
        timeF = 1  # 视频帧计数间隔频率
        #video_to_image(path,save_dir=path)
        #crop_image(path,save_dir=path,path=path)
        face_to_head_cheek(path,save_dir=path)


def crop_error_image():
    coordinate=['1.jpg']
    subject=['subject41']
    error=pd.read_csv("C:\RGB_data\DATASET_2\subject11\errorlist.csv")
    subject_path=os.path.join(dataset_dir,subject[0])
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_path=os.path.join(subject_path,'frame')
    gray_path = os.path.join(gray_path, coordinate[0])
    gray = cv2.imread(gray_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    image_dlr=os.path.join(subject_path,'frame')
    save_dlr="C:\RGB_data\error"

    a=0
    x, y, w, h=faces[a][0],faces[a][1],faces[a][2],faces[a][3]


    for i in range(101,2000):
        img=str(i)+'.jpg'
        image_path=os.path.join(image_dlr,img)
        image=cv2.imread(image_path)
        crop_img=image[y:y+h,x:x+w]
        save_path=os.path.join(save_dlr,img)
        cv2.imwrite(save_path,crop_img)











