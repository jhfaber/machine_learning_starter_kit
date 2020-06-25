import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

class TrainingDataImg:
    """ 
    Teniendo X.pickle, y.pickle
    train = TrainingDataImg()
    X,y = train.load_info()
    
    """

    def __init__(self,IMG_SIZE=100,DATADIR="/mnt/c/Users/JOHN/development/im_detection/PetImages"):
        
        self.IMG_SIZE=IMG_SIZE
        # DATADIR = "C:/Users/JOHN/development/im_detection/PetImages"
        self.DATADIR = DATADIR
        self.CATEGORIES = ["Dog", "Cat"]
        self.img_array = []
        self.training_data = []
        
        self.X= [] #DATA EN CRUDO
        self.y =[] #DATA OBJETIVO
    
    def create_training_data(self): 
        """Crea y procesa la data, IMG_SIZE resize la img deja todo en X,y 
        estructura normal: [PetImages / [Cat,Dog] ]
        input: self.DATADIR, self.CATEGORIES (categories tiene array de carpetas llenas de imagenes)
        output: X,y  [X-> data reducido a IMG_SIZE, y -> objetivo o clasificacion]
        """
        for category in self.CATEGORIES:  # do dogs and cats
            path = os.path.join(self.DATADIR,category)  # create path to dogs and cats
            class_num = self.CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

            for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))  # resize to normalize data size
                    self.training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
                
        #revuelve el training_data pues estan en orden primero (perro luego gatos)
        random.shuffle(self.training_data)
        
        
        for features,label in self.training_data:
            self.X.append(features)
            self.y.append(label)
        
        ###########################
        # RESHAPE TODA LA INFO
        ###################################
        self.X = np.array(self.X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
    
    def save_info(self):
        """input X,y
        output data guardada como X,y en DATADIR"""
        if(self.X != []):
            pickle_out = open(self.DATADIR + "/X.pickle","wb")
            pickle.dump(self.X, pickle_out)
            pickle_out.close()

            pickle_out = open(self.DATADIR+ "/y.pickle","wb")
            pickle.dump(self.y, pickle_out)
            pickle_out.close()
        else:
            print("Sin datos para guardar")
        
    def load_info(self):
        """
        input: DATADIR donde esta X,y
        output: arrays de X,y
        
        """
        pickle_in = open(self.DATADIR+"/X.pickle","rb")
        self.X = pickle.load(pickle_in)
        pickle_in = open(self.DATADIR + "/y.pickle","rb")
        self.y = pickle.load(pickle_in)
        
        return self.X, self.y
        
    def getImage(self):
        """obtiene la primera imagen del dataset """
        for category in self.CATEGORIES:  # do dogs and cats
            path = os.path.join(self.DATADIR,category)  # create path to dogs and cats
            for img in os.listdir(path):  # iterate over each image per dogs and cats
                self.img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                plt.imshow(self.img_array, cmap='gray')  # graph it
                plt.show()  # display!

                break  # we just want one for now so break
            break  #...and one more!
    
    def showImgResize(self):
        new_array = cv2.resize(self.img_array, (self.IMG_SIZE, self.IMG_SIZE))
        plt.imshow(new_array, cmap='gray')
        plt.show()
    def showImage(self):
        plt.imshow(self.img_array, cmap='gray')
        plt.show()
 
