import numpy as np
import os
import cv2
import pandas as pd
def UBFC_file_path():

    dataset_dlr="/home//lab70636//Desktop//DATASET_2"

    def sort_subject_dlr(subject_list):
        subject=[]
        count=0


        for i in subject_list:
            if (len(i)==8):
                if (count == 0):
                    num=3
                if (count == 1):
                    num=1
                if (count == 2):
                    num=4
                if (count == 3):
                    num=9
                if (count == 4):
                    num=8
                if (count == 5):
                    num=5

                i=i.rstrip(str(num))
                i=i+'0'+str(num)
                count += 1

            subject.append(i)

        subject.sort(key=lambda x: int(x[7:9]))
        count=1
        for i in range(0,6):
            num=str(count)
            subject[i]=subject[i].rstrip('0'+num)
            subject[i]=subject[i]+num

            count += 1
            if (count == 2):
                count += 1
            if (count == 6):
                count += 1
            if (count == 7):
                count += 1

        return subject







    subject_list=os.listdir(dataset_dlr)
    subject_list=sort_subject_dlr(subject_list)



    train_forehead_path=[]
    train_cheek_path=[]

    val_forehead_path=[]
    val_cheek_path=[]

    count=0
    img_count=600
    for i in subject_list:
        subject_path=os.path.join(dataset_dlr,i)
        forehead_dlr=os.path.join(subject_path,"head")
        cheek_dlr=os.path.join(subject_path,"cheek")

        forehead_list=os.listdir(forehead_dlr)



        cheek_list=os.listdir(cheek_dlr)
        if(count<35):
            for j in range(0,len(forehead_list)-img_count):
                eighty_stack=[]
                for k in range(j,j+img_count):
                    k=str(k)+'.jpg'
                    k=os.path.join(forehead_dlr,k)
                    eighty_stack.append(k)
                train_forehead_path.append(eighty_stack)

            for j in range(0,len(cheek_list)-img_count):
                eighty_stack=[]
                for k in range(j,j+img_count):
                    k=str(k)+'.jpg'
                    k=os.path.join(cheek_dlr,k)
                    eighty_stack.append(k)
                train_cheek_path.append(eighty_stack)
        else:
            for j in range(0,len(forehead_list)-img_count):
                eighty_stack = []
                for k in range(j, j + img_count):
                    k = str(k) + '.jpg'
                    k = os.path.join(forehead_dlr, k)
                    eighty_stack.append(k)
                val_forehead_path.append(eighty_stack)

            for j in range(0,len(cheek_list)-img_count):
                eighty_stack = []
                for k in range(j,j+img_count):
                    k =str(k) + '.jpg'
                    k=os.path.join(cheek_dlr,k)
                    eighty_stack.append(k)
                val_cheek_path.append(eighty_stack)

        count=count+1
        print(count)



    train_forehead_path=np.array(train_forehead_path)
    np.save('/home/lab70636/Desktop/RGB/program/save_npy/all_train_forehead.npy',train_forehead_path)


    train_cheek_path=np.array(train_cheek_path)
    np.save( '/home/lab70636/Desktop/RGB/program/save_npy/all_train_cheek.npy',train_cheek_path)


    val_forehead_path=np.array(val_forehead_path)
    np.save('/home/lab70636/Desktop/RGB/program/save_npy/all_val_forehead.npy',val_forehead_path )


    val_cheek_path=np.array(val_cheek_path)
    np.save( '/home/lab70636/Desktop/RGB/program/save_npy/all_val_cheek.npy',val_cheek_path)


def cohface_file_path():

    dataset_dlr = "/home/lab70636/Desktop/coface/cohface"
    train_forehead_path = []
    train_cheek_path = []

    val_forehead_path = []
    val_cheek_path = []

    count = 1
    img_count = 256
    train_shifting=256
    val_shifting=50
    all_person_path = os.listdir(dataset_dlr)


    for i in range(1,39):
        if (count==24):
            count+=1
        elif (count==26):
            count+=1
        person_path = os.path.join(dataset_dlr, str(count))
        person_video_path = os.listdir(person_path)

        for r in range(0,len(person_video_path)):

            subject_path = os.path.join(person_path, str(r))
            print("subject_path=",subject_path)
            forehead_dlr = os.path.join(subject_path, "head")
            cheek_dlr = os.path.join(subject_path, "cheek")

            forehead_list = os.listdir(forehead_dlr)
            cheek_list = os.listdir(cheek_dlr)
            if (count <= 28):
                for j in range(2, len(forehead_list) - img_count,train_shifting):
                    eighty_stack = []
                    for k in range(j, j + img_count):
                        k = str(k) + '.jpg'
                        k = os.path.join(forehead_dlr, k)
                        eighty_stack.append(k)
                    train_forehead_path.append(eighty_stack)

                for j in range(2, len(cheek_list) - img_count,train_shifting):
                    eighty_stack = []
                    for k in range(j, j + img_count):
                        k = str(k) + '.jpg'
                        k = os.path.join(cheek_dlr, k)
                        eighty_stack.append(k)
                    train_cheek_path.append(eighty_stack)
                print("train:{}".format(count))
            else:
                for j in range(2, len(forehead_list) - img_count,val_shifting):
                    eighty_stack = []
                    for k in range(j, j + img_count):
                        k = str(k) + '.jpg'
                        k = os.path.join(forehead_dlr, k)
                        eighty_stack.append(k)
                    val_forehead_path.append(eighty_stack)

                for j in range(2, len(cheek_list) - img_count,val_shifting):
                    eighty_stack = []
                    for k in range(j, j + img_count):
                        k = str(k) + '.jpg'
                        k = os.path.join(cheek_dlr, k)
                        eighty_stack.append(k)
                    val_cheek_path.append(eighty_stack)
                print("val:{}".format(count))
        count = count + 1






    train_forehead_path = np.array(train_forehead_path)
    np.save('/home/lab70636/Desktop/RGB/program/save_npy/cohface_train_forehead.npy', train_forehead_path)

    train_cheek_path = np.array(train_cheek_path)
    np.save('/home/lab70636/Desktop/RGB/program/save_npy/cohface_train_cheek.npy', train_cheek_path)

    val_forehead_path = np.array(val_forehead_path)
    np.save('/home/lab70636/Desktop/RGB/program/save_npy/cohface_val_forehead.npy', val_forehead_path)

    val_cheek_path = np.array(val_cheek_path)
    np.save('/home/lab70636/Desktop/RGB/program/save_npy/cohface_val_cheek.npy', val_cheek_path)

def cohface_file_path_new():
    dataset_dlr = "E:\coface\cohface"
    train_forehead_path = []
    train_cheek_path = []
    train=[]
    val_forehead_path = []
    val_cheek_path = []
    val=[]
    count = 1
    img_count = 256

    all_person_path = os.listdir(dataset_dlr)

    for i in range(1, 39):
        if (count == 24):
            count += 1
        elif (count == 26):
            count += 1
        person_path = os.path.join(dataset_dlr, str(count))
        person_video_path = os.listdir(person_path)

        for r in range(0, len(person_video_path)):
            subject_path = os.path.join(person_path, str(r))
            print("subject_path=", subject_path)
            if(count<=28):

                train.append(subject_path)
            else:
                val.append(subject_path)

        count = count + 1

    train = np.array(train)
    np.save('C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_train.npy', train)

    val = np.array(val)
    np.save('C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_val.npy', val)

    #val= np.array(val)
    #np.save('/home/lab70636/Desktop/RGB/program/save_npy/cohface_val.npy', val)

    #val_cheek_path = np.array(val_cheek_path)
    # np.save('/home/lab70636/Desktop/RGB/program/save_npy/cohface_val_cheek.npy', val_cheek_path)

cohface_file_path_new()