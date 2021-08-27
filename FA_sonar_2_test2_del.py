######sonar######

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import math
import random
from tensorflow.keras.models import Sequential
#from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, BatchNormalization
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras import optimizers
from keras import callbacks
from keras.callbacks import EarlyStopping
import time
import gc
#from  multiprocessing import Process, Pool
# from numba import cuda
# print(cuda.gpus)
#pool = Pool(8)
start = time.clock()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# 自動增長 GPU 記憶體用量
import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# config=tf.compat.v1.ConfigProto() 
# config.gpu_options.visible_device_list = '0' 
# config.gpu_options.allow_growth = True 
# sess=tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #選擇哪一塊gpu
config = ConfigProto()
config.allow_soft_placement=True #如果你指定的設備不存在，允許TF自動分配設備
config.gpu_options.per_process_gpu_memory_fraction=0.7  #分配百分之七十的顯存給程序使用，避免內存溢出，可以自己調整
config.gpu_options.allow_growth = True   #按需分配顯存，這個比較重要
session = InteractiveSession(config=config)
#sess = tf.Session(config=config)
#sess.run(tf.global_variables_initializer())
#tf.graph.finalize()
tf.compat.v1.get_default_graph()
#with tf.device('gpu'):
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 

def train_test_split1(x, y):
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test

def standard_scaler(x_train, x_test):
    sc = StandardScaler()
    x_train = x_train.values
    x_test = x_test.values
    x_scale_train = sc.fit(x_train)
    x_scale_test = sc.fit(x_test)
    x_train = x_scale_train.transform(x_train)
    x_test = x_scale_train.transform(x_test)
    return x_train, x_test

def normalization(x_train, x_test, y_train, y_test):
    normalizer=Normalizer()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.fit_transform(x_test)
    y_train = normalizer.fit_transform(y_train)
    y_test = normalizer.fit_transform(y_test)
    return x_train, x_test, y_train, y_test

def NN_model_structure_classification(x_train, y_train, size, epochs, x_test, y_test):
    tf.compat.v1.get_default_graph()
    #np.random.seed(3)
    # 設定 Keras 使用的 Session
    #tf.compat.v1.keras.backend.set_session(sess)
    model = Sequential()
    model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = (x_train.shape[1])))
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    #history = model.fit(x_train, y_train, batch_size = size, epochs = epochs, shuffle = True, validation_data=(x_test, y_test), callbacks=[callback])
    history = model.fit(x_train, y_train, batch_size = size, epochs = epochs, shuffle = True, validation_data=(x_test, y_test))
    history_dict = history.history
    history_dict.keys()
    acc_values = history_dict['accuracy'] 
    max_acc = max(acc_values) 
    return model, history, max_acc
##############################################################
######################  Import Dataset  ######################
##############################################################
header_list=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 
             'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 
             'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 
             'f26', 'f27', 'f28', 'f29', 'f30','f31', 'f32', 'f33', 
             'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 
             'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 
             'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 
             'f58', 'f59', 'f60', 'Label']

df_sonar=pd.read_csv('dataset\\sonar.all-data', header=None)
    
df_sonar.columns=header_list
# convert string to number
le = LabelEncoder()

string_sonar = df_sonar[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30','f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'Label']]
df_sonar.dtypes
df_sonar[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34','f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'Label']] = string_sonar.apply(le.fit_transform)
df_sonar[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'Label']] = df_sonar[[    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'Label']].astype(np.int64)
print(df_sonar)
df_sonar['Label']
x_sonar = df_sonar.iloc[:, 0:60]
y_sonar = df_sonar['Label']
#######################################################################################################################
########################################### Firefly Algorithm #########################################################
#######################################################################################################################
def initial_sol(solutions, num_of_features):
    feature_index_list = []
    encode_list = []
    encode_binary_list=[]
    feature_index=[]
    encode_binary=[]
    threshold=0.5
    np.random.seed(20)
    for i in range(solutions):
        encode = np.random.uniform(0,1, size=num_of_features)
        encode_binary=np.copy(encode)
        encode_binary[encode_binary > 0.5]=1
        encode_binary[encode_binary <= 0.5]=0
        check = np.sum(encode_binary)
        while check == 0:
            encode = np.random.uniform(0,1,size=num_of_features)
            encode_binary = np.copy(encode)
            encode_binary[encode_binary > 0.5]=1
            encode_binary[encode_binary <= 0.5]=0
            check = np.sum(encode_binary)
        encode_list.append(encode)
        encode_binary_list.append(encode_binary)
        #I=np.where(encode>threshold)
        #feature_index=I[0].T
        for j in range(len(encode)):
            if encode[j] > threshold:
                feature_index.append(j)
        feature_index_list.append(feature_index)
        feature_index=[]
        del check, encode, encode_binary
        gc.collect()
    return encode_list, encode_binary_list, feature_index_list 

def euclidean_distance(encode1, encode2):
    distance=0
    for i in range(60):
        distance1 = np.linalg.norm(np.array(encode2)-np.array(encode1))
        distance += distance1
        del distance1
        #gc.collect()
    return distance

def move_Xj_towards_Xk(a,βo, encode1, encode2):
    rjk = euclidean_distance(encode1, encode2)
    γrjk = -0.1*(rjk**2)
    #β = βo*math.exp(γrjk)
    β = βo*np.exp(γrjk)
    new_firefly = np.array(encode1)+np.array(β*(np.array(encode2)-np.array(encode1)))+np.array(a*np.array(np.random.uniform(low = -0.5, high = 0.5, size=(x_sonar.shape[1]))))
    new_firefly[new_firefly < 0]=0
    new_firefly[new_firefly > 1]=1
    new_firefly_binary=np.copy(new_firefly)
    new_firefly_binary[new_firefly_binary > 0.5]=1
    new_firefly_binary[new_firefly_binary <= 0.5]=0
    del rjk,γrjk, βo,β
    #gc.collect()
    return new_firefly, new_firefly_binary

def particle_best(feature_index_list, encode_list, encode_binary_list):   #一開始全部未經select的feature放進去MLP求accuracy
    
    firefly_result_list=[]
    #model_list=[]
    a=[]
    for i in range(len(feature_index_list)):
        x = df_sonar.iloc[:, feature_index_list[i]]
        y = df_sonar['Label']
        # labelencoder_Y = LabelEncoder()
        # y = labelencoder_Y.fit_transform(y)
        x_train, x_test, y_train, y_test = train_test_split1(x, y)
        x_train, x_test = standard_scaler(x_train, x_test)
        model, history, max_acc = NN_model_structure_classification(x_train, y_train, 1, 10, x_test, y_test)
        train=model.evaluate(x_train, y_train)
        test=model.evaluate(x_test, y_test)
        #model_list.append(model)
        a.append(train[1])
        a.append(test[1])
        firefly_result_list.append(a)  
        a=[]
        del model,history,max_acc,train, test 
    return feature_index_list, encode_list, encode_binary_list, firefly_result_list#, model_list

def fitness_of_new_firefly(new_firefly, new_firefly_binary, x, y):    #X_val, Y_val
    
    firefly_result_list=[]
    threshold=0.5
    feature_index=[]
    # check = np.sum(new_firefly_binary)
    # if check ==0:
    #     new_firefly = np.random.uniform(0,1, size=len(new_firefly))
    #     new_firefly_binary = np.copy(new_firefly)
    #     new_firefly_binary[new_firefly_binary > 0.5]=1
    #     new_firefly_binary[new_firefly_binary <= 0.5]=0
    
    for i in range(len(new_firefly)):
        if new_firefly[i] > threshold:
            feature_index.append(i)   
            #print('feature_index_fitness:',feature_index)
    x_fs=x.iloc[:,feature_index]
    x_train, x_test, y_train, y_test = train_test_split1(x_fs, y)
    x_train, x_test = standard_scaler(x_train, x_test)
    model, history, max_acc = NN_model_structure_classification(x_train, y_train, 1, 10, x_test, y_test)
    train=model.evaluate(x_train, y_train)
    test=model.evaluate(x_test, y_test)
    firefly_result_list.append(train[1])
    firefly_result_list.append(test[1])
    del model,history,max_acc,train,test 
    return new_firefly, new_firefly_binary, feature_index, firefly_result_list#, model
    
def global_best(encode_list, encode_binary_list, firefly_result_list, feature_index_list):      
    #compare=[-math.inf, -math.inf]
    compare=[-math.inf]
    for i in range(len(firefly_result_list)):
        #if firefly_result_list[i][0] > compare[0]:
        if firefly_result_list[i] > compare:
            compare = firefly_result_list[i]
            gbest_result = compare
            gbest, gbest_binary = encode_list[i], encode_binary_list[i]
            gbest_feature_index=feature_index_list[i]
            #del compare
            #gbest_model=model_list[i]
    return gbest, gbest_binary, gbest_result, gbest_feature_index#, gbest_model

def onepointchange(new_firefly, new_firefly_binary, num_of_features):
   
   
   position = np.random.randint(0,num_of_features)
   if new_firefly_binary[position] == 1:
       new_firefly_binary[position] = 0
       new_firefly[position]=np.random.uniform(0,0.5)
       check = np.sum(new_firefly_binary)
       while check == 0:
           position = np.random.randint(0,num_of_features)
           new_firefly_binary[position]= 1
           new_firefly[position] = np.random.uniform(0.5001,1)
           check = np.sum(new_firefly_binary)
        
   else:
        new_firefly_binary[position] = 1
        new_firefly[position]=np.random.uniform(0.5001,1)
        
        

   del position
        #gc.collect()
   return  new_firefly, new_firefly_binary

def twopointchange(new_firefly, new_firefly_binary, num_of_features):
    position=[]
    
   
        #position = np.random.randint(low=0, high=len(new_firefly_binary), size=2)
    position = np.random.choice(60, 2, replace = False)
    position1 = position[0]
    position2 = position[1]
    if new_firefly_binary[position1]==1 and new_firefly_binary[position2]==1:
        new_firefly_binary[position1]=0
        new_firefly_binary[position2]=0
        new_firefly[position1] = np.random.uniform(0,0.5)
        new_firefly[position2] = np.random.uniform(0,0.5)
        check = np.sum(new_firefly_binary)
        while check == 0:
            position = np.random.randint(0,num_of_features)
            new_firefly_binary[position]= 1
            new_firefly[position] = np.random.uniform(0.5001,1)
            check = np.sum(new_firefly_binary)
            
    elif new_firefly_binary[position1]==0 and new_firefly_binary[position2]==0:
        new_firefly_binary[position1]=1
        new_firefly_binary[position2]=1
        new_firefly[position1] = np.random.uniform(0.5001,1)
        new_firefly[position2] = np.random.uniform(0.5001,1)
    elif new_firefly_binary[position1]==0 and new_firefly_binary[position2]==1:
        new_firefly_binary[position1]=1
        new_firefly_binary[position2]=0
        new_firefly[position1] = np.random.uniform(0.5001,1)
        new_firefly[position2] = np.random.uniform(0,0.5)
    elif new_firefly_binary[position1]==1 and new_firefly_binary[position2]==0:
        new_firefly_binary[position1]=0
        new_firefly_binary[position2]=1
        new_firefly[position1] = np.random.uniform(0,0.5)
        new_firefly[position2] = np.random.uniform(0.5001,1)
        
    del position,position1,position2
        #gc.collect()
        
        # while position2==position1:
        #     position2 = random.randint(0,num_of_features-1)
    return new_firefly, new_firefly_binary
        

def shaking(new_firefly, new_firefly_binary, num_of_features):

  count=len(new_firefly)
  
  k = np.random.uniform(0.5,1)
  times=round(count*k)  

  for i in range(times):
      position_shaking = np.random.randint(0,len(new_firefly))
      for k in range(len(new_firefly_binary)):
               if new_firefly_binary[position_shaking] == 1:
                  new_firefly_binary[position_shaking] = 0
                  new_firefly[position_shaking]=np.random.uniform(0,0.4999)
               else:
                   new_firefly_binary[position_shaking] = 1
                   new_firefly[position_shaking]=np.random.uniform(0.5001,1)         
               check = np.sum(new_firefly_binary)
               if check == 0:
                   while check == 0:
                       position = np.random.randint(0,num_of_features)
                       new_firefly_binary[position]= 1
                       new_firefly[position] = np.random.uniform(0.5001,1)
                       check = np.sum(new_firefly_binary)
      del position_shaking
  del count,k,times,check
  #gc.collect()
                       
  return new_firefly, new_firefly_binary

def FA_Algorithm(encode_list, encode_binary_list, firefly_result_list, feature_index_list, x, y, current_best_encode, current_best_binary, current_best_result, current_best_feature_subset):
    #FA Algorithm
    iteration=5
    a=1
    βo=1
    rand1_list=np.random.rand(20)
    rand2_list=np.random.rand(20)
    
    for i in range(iteration):
        print('iteration:', i)
        a=1
        if rand2_list[i]<0.5:
            βo=rand1_list[i]
        else:
            βo=βo

        length=len(firefly_result_list)
        for j in range(length):
            
            for k in range(length):

                if firefly_result_list[k]>firefly_result_list[j]:
                    new_firefly, new_firefly_binary = move_Xj_towards_Xk(a,βo, encode_list[j], encode_list[k])
    
            encode_list[j], encode_binary_list[j], feature_index_list[j], firefly_result_list[j] = fitness_of_new_firefly(new_firefly, new_firefly_binary, x, y)
            #del rand1_list, rand2_list
            gc.collect()                   
                    
                                    ##########################VNS##########################
#global best
        best_VNS_encode = encode_list.copy()
        best_VNS_encode_binary = encode_binary_list.copy()
        best_VNS_feature_index_list = feature_index_list.copy()
        best_VNS_result_list = firefly_result_list.copy()
        #best_VNS_model_list = model_list.copy()

        
        for j in range(length):
            print('iteration',i)
            print('firefly_result_list:',j)
            #current best
            encode_VNS = best_VNS_encode[j].copy()
            encode_VNS_binary = best_VNS_encode_binary[j].copy()
            #local best
            old_VNS_encode = best_VNS_encode[j].copy()
            old_VNS_encode_binary = best_VNS_encode_binary[j].copy()
            for l in range(10):
                print('VNS_iteration:',l)
                encode_VNS, encode_VNS_binary = shaking(old_VNS_encode.copy(), old_VNS_encode_binary.copy(), 60)
                #print('shaking_encode:',encode_VNS)
                #print('shaking_encode_binary:',encode_VNS_binary)
                encode_VNS, encode_VNS_binary, feature_index_encode_vns, encode_vns_result = fitness_of_new_firefly(encode_VNS, encode_VNS_binary, x, y)        
                
                old_VNS_encode = encode_VNS.copy()
                old_VNS_encode_binary = encode_VNS_binary.copy()
                old_feature_index = feature_index_encode_vns.copy()
                old_VNS_result = encode_vns_result.copy()
                #old_model = model_vns
          #new:  
                count_update=0
                for m in range(2):
                    print('m',m)
                    for n in range(10): 
                        if m==0:
                             encode_VNS, encode_VNS_binary = onepointchange(old_VNS_encode.copy(), old_VNS_encode_binary.copy(), 60)
                             #print('VNS1_encode:',encode_VNS)
                             #print('VNS1_encode_binary:',encode_VNS_binary)
                        else:
                             encode_VNS, encode_VNS_binary = twopointchange(old_VNS_encode.copy(), old_VNS_encode_binary.copy(), 60)
                             #print('VNS2_encode:',encode_VNS)
                             #print('VNS2_encode_binary:',encode_VNS_binary)
                        encode_VNS, encode_VNS_binary, feature_index_encode_VNS, VNS_result = fitness_of_new_firefly(encode_VNS, encode_VNS_binary, x, y)  
                        
                        # compare with " neighborhood solution " and " shaking solution "
                        #print('VNS result', VNS_result)
                        #print('Old VNS result', old_VNS_result)
                        if VNS_result[1]>old_VNS_result[1]:
                            #print('change VNS')
                            old_VNS_encode = encode_VNS.copy()
                            old_VNS_encode_binary = encode_VNS_binary.copy()
                            old_feature_index = feature_index_encode_VNS.copy() 
                            old_VNS_result = VNS_result.copy()
                            #old_model = model_encode_VNS
                            #del encode_VNS.copy(),encode_VNS_binary.copy(),feature_index_encode_VNS.copy(),VNS_result.copy()
                            #gc.collector()
                            #print('gc is working_1')
                            count_update = count_update+1
            #new:改成 
                    if m==1 and count_update>=1:
                        m=-1 
                        count_update=0
                    gc.collect()
                # compare with " neighborhood solution " and " FA solution "
                #print('old_VNS_result:',old_VNS_result[1])
                #print('best_VNS_result_list[j]:',best_VNS_result_list[j][1])
                if old_VNS_result[1] > best_VNS_result_list[j][1]:   
                     
                    best_VNS_encode[j] = old_VNS_encode.copy()
                    best_VNS_encode_binary[j] = old_VNS_encode_binary.copy()
                    best_VNS_feature_index_list[j] = old_feature_index.copy()
                    best_VNS_result_list[j] = old_VNS_result.copy()
                    
                    
                    
                    
                    
                    #best_VNS_model_list[j] = old_model
                    #print('Result:',best_VNS_result_list[j])             
            
        ##############REF:亦庭############### 
        #########VNS Solution for FA#########
#new:往前位移四格到上一個for l in range
            encode_list[j] = best_VNS_encode[j].copy()
            encode_binary_list[j] = best_VNS_encode_binary[j].copy()
            feature_index_list[j] = best_VNS_feature_index_list[j].copy()
            firefly_result_list[j]=best_VNS_result_list[j].copy()
            #del best_VNS_encode[j],best_VNS_encode_binary[j],best_VNS_feature_index_list[j],best_VNS_result_list[j]
            #print('gc is working_2')
            gc.collect()
            
            
            #model_list[j] = best_VNS_model_list[j]
        #####################################

        for p in range(len(firefly_result_list)):
       
#(6)在哪邊有current_best,不要亂寫,那是best_vns_firefly_result_list[j][1],出了vns,要先把best_vns還回給原先的firefly解,
            if firefly_result_list[p][1] > current_best_result[1]:
                current_best_result = firefly_result_list[p].copy()
                current_best_feature_subset = feature_index_list[p].copy()
                current_best_encode = encode_list[p].copy()
                current_best_binary = encode_binary_list[p].copy()
                #current_best_model = model_list[p]
                #del firefly_result_list[p],feature_index_list[p],encode_list[p],encode_binary_list[p]
                #print('gc is working_3')
                gc.collect()
                
                
    return current_best_encode, current_best_binary, current_best_result, current_best_feature_subset#, current_best_model


encode_list, encode_binary_list, feature_index_list= initial_sol(20, 60)
feature_index_list, encode_list, encode_binary_list, firefly_result_list = particle_best(feature_index_list, encode_list, encode_binary_list)
current_best_encode, current_best_binary, current_best_result, current_best_feature_subset= global_best(encode_list, encode_binary_list, firefly_result_list, feature_index_list)
best_encode, best_binary, best_result, best_feature = FA_Algorithm(encode_list, encode_binary_list, firefly_result_list, feature_index_list, x_sonar, y_sonar, current_best_encode, current_best_binary, current_best_result, current_best_feature_subset)

print('best_encode', best_encode)
print('best_result', best_result)
print('best_feature', best_feature)
elapsed = (time.clock() - start)
print("Time used:",elapsed)

f = open('sonar_vns_prof_test_lasttest_new(wz GPU_memory_setting)_iter_5_VNS_iter10_converge_10_BS_1_epochs_10_NP_20_RS_20_a_1_b_m2_g_0.1_threshold_0.5.txt', 'w')
f.write("Feature subset:\n")
f.write(str(best_feature))
f.write("Accuracy:\n")
f.write(str(best_result))
f.write("Encode:\n")
f.write(str(best_encode))
f.write("Time used:\n")
f.write(str(elapsed))
f.close()
    