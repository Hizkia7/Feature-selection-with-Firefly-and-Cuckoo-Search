######breast_cancer######

import numpy as np
import pandas as pd
import csv
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
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
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
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)
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

#Data Preprocessing    
header_list=['class','a1','a2','a3','a4','a5','a6','id']

with open('dataset\\monks-2.test', 'r') as f:
    df_monk2_row_test = [line.split() for line in f]

df_monk2=pd.DataFrame(data=df_monk2_row_test, columns=header_list)
print(df_monk2)
    
#avoid missing data
df_monk2=df_monk2.dropna()
df_monk2[['class','a1','a2','a3','a4','a5','a6']]=df_monk2[['class','a1','a2','a3','a4','a5','a6']].astype(np.int64)
print(df_monk2.dtypes)


df_monk2_final=df_monk2.drop(['id','class'], axis=1)

#######################################################################################################################
########################################### Cuckoo Search Algorithm #########################################################
#######################################################################################################################


def initial_sol(solution, total_features):
    
    np.random.seed(7)
    encode_list=[]
    encode_binary_list=[]
    feature_subset_list=[]
    feature_index=[]
    threshold=0.5
    
    for i in range(solution):
        encode=np.random.uniform(0,1, size=total_features)
        encode_binary=np.copy(encode)
        encode_binary[encode_binary>=0.5]=1
        encode_binary[encode_binary<0.5]=0
        total_encode=np.sum(encode_binary)
        
        #check whether the all of the binary encode is 0
        while total_encode == 0:
            encode=np.random.uniform(0,1, size=total_features)
            encode_binary=np.copy(encode)
            encode_binary=encode_binary[encode_binary>=0.5]=1
            encode_binary=encode_binary[encode_binary<0.5]=0
            total_encode=np.sum(encode_binary)
        
        #extract feature index
        for i in range(len(encode)):
            if encode[i]>=0.5:
                feature_index.append(i)
        
        #record encode, encode binary and feature index        
        encode_list.append(encode)
        encode_binary_list.append(encode_binary)
        feature_subset_list.append(feature_index)
        feature_index=[]
    
    return encode_list, encode_binary_list, feature_subset_list
        
def particle_best(encode_list, encode_binary_list, feature_subset_list):   #一開始全部未經select的feature放進去MLP求accuracy
    
    cuckoo_result_list=[]
    #model_list=[]
    a=[]
    for i in range(len(feature_subset_list)):
        x = df_monk2_final.iloc[:, feature_subset_list[i]]
        y = df_monk2['class']
        x_train, x_test, y_train, y_test = train_test_split1(x, y)
        x_train, x_test = standard_scaler(x_train, x_test)
        model, history, max_acc = NN_model_structure_classification(x_train, y_train, 32, 50, x_test, y_test)
        print('evaluate')
        train=model.evaluate(x_train, y_train)
        test=model.evaluate(x_test, y_test)
        #model_list.append(model)
        a.append(train[1])
        a.append(test[1])
        cuckoo_result_list.append(a)  
        a=[]
        #del model,history,max_acc,train
        
    return encode_list, encode_binary_list, feature_subset_list, cuckoo_result_list #model_list

    
def global_best(encode_list, encode_binary_list, feature_subset_list, cuckoo_result_list):      
    
    #compare=[-math.inf, -math.inf]
    compare=[-math.inf]
    for i in range(len(cuckoo_result_list)):
        #if firefly_result_list[i][0] > compare[0]:
        if cuckoo_result_list[i] > compare:
            compare = cuckoo_result_list[i]
            gbest_result = compare
            gbest, gbest_binary = encode_list[i], encode_binary_list[i]
            gbest_feature_subset=feature_subset_list[i]
            #del compare
            #gbest_model=model_list[i]
    
    return gbest, gbest_binary, gbest_feature_subset, gbest_result#, gbest_model


def update_nest(alpha0, random_encode_i, encode_list):
    
    new_feature_subset=[]
    
    random_encode_j=encode_list[np.random.choice(len(encode_list),replace=False)]
    alpha=np.kron(alpha0,(np.array(random_encode_j)-np.array(random_encode_i)))
    #λ=np.random.uniform(1,3)
    λ=1.5
    Γλ=math.gamma(λ)
    π=math.pi
    σ_square=((math.gamma(1+λ)*math.sin((π*λ)/2))/(math.gamma((1+λ)/2)*λ*2**((λ-1)/2)))**1/λ
    u=np.random.uniform(0, σ_square)
    v=np.random.uniform(0,1)
    s=u*(abs(v)**(-1/λ))
    Levy_1=((λ*Γλ*math.sin((π*λ)/2))/π)
    Levy_2=(1/(s**(1+λ)))
    Lévy_λ=((λ*Γλ*math.sin((π*λ)/2))/π)*np.real(1/(s**(1+λ)))
    print('Levy_1', Levy_1)
    print('Levy_2', Levy_2)
    print('s', s)
    print('σ_square', σ_square)
    print('encode_i', random_encode_i)
    print('encode_j', random_encode_j)
    print('np array encode_i', np.array(random_encode_i))
    print('np array encode_j', np.array(random_encode_j))
    print('encode_i shape', np.array(random_encode_i).shape)
    print('encode_j shape', np.array(random_encode_j).shape)
    print('minus',np.array(random_encode_j)-np.array(random_encode_i))
    print('Levy', Lévy_λ)
    print('alpha', alpha)
    #getting new encode
    print('encode', np.array(random_encode_i))
    print('kron', np.kron(alpha,Lévy_λ))
    new_encode=np.array(random_encode_i)+np.kron(alpha,Lévy_λ)
    new_encode[new_encode>1]=1
    new_encode[new_encode<0]=0
    new_encode_binary=np.copy(new_encode)
    new_encode_binary[new_encode_binary>=0.5]=1
    new_encode_binary[new_encode_binary<0.5]=0   
    #extract feature index
    for i in range(len(new_encode)):
        if new_encode[i]>=0.5:
            new_feature_subset.append(i)
    
    return new_encode, new_encode_binary, new_feature_subset

def fitness_of_new_nest(new_encode, new_encode_binary, new_feature_subset):
    new_cuckoo_result=[]
    
    #model.set_weights(new_nest)
    x = df_monk2_final.iloc[:, new_feature_subset]
    y = df_monk2['class']
    x_train, x_test, y_train, y_test = train_test_split1(x, y)
    x_train, x_test = standard_scaler(x_train, x_test)
    model, history, max_acc = NN_model_structure_classification(x_train, y_train, 32, 50, x_test, y_test)
    train=model.evaluate(x_train, y_train)
    test=model.evaluate(x_test, y_test)
    #model_list.append(model)
    new_cuckoo_result.append(train[1])
    new_cuckoo_result.append(test[1])  

    return new_encode, new_encode_binary, new_feature_subset, new_cuckoo_result

def fraction_Pa_nest(Pa, alpha0, random_encode_i, encode_list):
    new_feature_subset_Pa=[]
    
    random_encode_j=encode_list[np.random.choice(len(encode_list),replace=False)]
    
    equality = np.array_equal(random_encode_j, random_encode_i)
    print('equality', equality)
    while equality == True:
        random_encode_j=encode_list[np.random.choice(len(encode_list),replace=False)]
        equality = np.array_equal(random_encode_j, random_encode_i)
        
    random_encode_k=encode_list[np.random.choice(len(encode_list),replace=False)]
    
    equality1 = np.array_equal(random_encode_k, random_encode_i)
    equality2 = np.array_equal(random_encode_k, random_encode_j)
    print('equality1', equality1)
    print('equality2', equality2)
    while equality1 == True or equality2 == True:
        random_encode_k=encode_list[np.random.choice(len(encode_list),replace=False)]
        equality1 = np.array_equal(random_encode_k, random_encode_i)
        equality2 = np.array_equal(random_encode_k, random_encode_j)
        
    print('random', random_encode_i)
    r=np.random.normal(0,1)
    H=np.heaviside(Pa,r)
    H_kron=np.kron(alpha0,H)
    minus=np.array(random_encode_j)-np.array(random_encode_k)
    print('random_encode_j', random_encode_j)
    print('random_encode_j', np.array(random_encode_j))
    print('random_encode_k', random_encode_k)
    print('random_encode_k', np.array(random_encode_k))
    print('minus', minus)
    #develop new encode
    encode_updater=np.array(np.kron(np.kron(alpha0,H),(np.array(random_encode_j)-np.array(random_encode_k))))
    new_encode_Pa=np.array(random_encode_i)+encode_updater
    new_encode_Pa
    new_encode_Pa[new_encode_Pa > 1]=1
    new_encode_Pa[new_encode_Pa < 0]=0
    new_encode_binary_Pa=np.copy(new_encode_Pa)
    new_encode_binary_Pa[new_encode_binary_Pa >= 0.5]=1
    new_encode_binary_Pa[new_encode_binary_Pa < 0.5]=0
    
    #extract feature index
    for i in range(len(new_encode_Pa)):
        if new_encode_Pa[i]>=0.5:
            new_feature_subset_Pa.append(i)
    
    return new_encode_Pa, new_encode_binary_Pa, new_feature_subset_Pa


def onepointchange(new_encode, new_encode_binary):
   
   feature_subset=[]
    
   #update encode and encode binary
   position = np.random.randint(0, len(new_encode))
   if new_encode_binary[position] == 1:
       new_encode_binary[position] = 0
       new_encode[position]=np.random.uniform(0,0.4999)
       check = np.sum(new_encode_binary)
       while check == 0:
           position = np.random.randint(0, len(new_encode))
           new_encode_binary[position]= 1
           new_encode[position] = np.random.uniform(0.5,1)
           check = np.sum(new_encode_binary)
        
   else:
        new_encode_binary[position] = 1
        new_encode[position]=np.random.uniform(0.5,1)
   
   #feature index extraction
   for i in range(len(new_encode)):
       if new_encode[i] >= 0.5:
           feature_subset.append(i)
        
   #del position
        #gc.collect()
   return  new_encode, new_encode_binary, feature_subset

def twopointchange(new_encode, new_encode_binary):
    feature_subset=[]
   
    #update encode and encode binary
    position = np.random.choice(6, 2, replace = False)
    position1 = position[0]
    position2 = position[1]
    if new_encode_binary[position1]==1 and new_encode_binary[position2]==1:
        new_encode_binary[position1]=0
        new_encode_binary[position2]=0
        new_encode[position1] = np.random.uniform(0,0.4999)
        new_encode[position2] = np.random.uniform(0,0.4999)
        check = np.sum(new_encode_binary)
        while check == 0:
            position = np.random.randint(0, len(new_encode))
            new_encode_binary[position]= 1
            new_encode[position] = np.random.uniform(0.5,1)
            check = np.sum(new_encode_binary)
            
    elif new_encode_binary[position1]==0 and new_encode_binary[position2]==0:
        new_encode_binary[position1]=1
        new_encode_binary[position2]=1
        new_encode[position1] = np.random.uniform(0.5,1)
        new_encode[position2] = np.random.uniform(0.5,1)
    elif new_encode_binary[position1]==0 and new_encode_binary[position2]==1:
        new_encode_binary[position1]=1
        new_encode_binary[position2]=0
        new_encode[position1] = np.random.uniform(0.5,1)
        new_encode[position2] = np.random.uniform(0,0.4999)
    elif new_encode_binary[position1]==1 and new_encode_binary[position2]==0:
        new_encode_binary[position1]=0
        new_encode_binary[position2]=1
        new_encode[position1] = np.random.uniform(0,0.4999)
        new_encode[position2] = np.random.uniform(0.5,1)
    
    # feature index extraction
    for i in range(len(new_encode)):
       if new_encode[i] >= 0.5:
           feature_subset.append(i)
    
    #del position,position1,position2
        #gc.collect()
        
        # while position2==position1:
        #     position2 = random.randint(0,num_of_features-1)
    return new_encode, new_encode_binary, feature_subset
        

def shaking(new_encode, new_encode_binary):

  feature_subset=[]  
  count=len(new_encode)
  
  k = np.random.uniform(0.5,1)
  times=round(count*k)  

  #update encode and encode binary
  for i in range(times):
      for k in range(len(new_encode_binary)):
          position_shaking = np.random.randint(0,len(new_encode))
          if new_encode_binary[position_shaking] == 1:
              new_encode_binary[position_shaking] = 0
              new_encode[position_shaking]=np.random.uniform(0,0.4999)
          else:        
              new_encode_binary[position_shaking] = 1
              new_encode[position_shaking]=np.random.uniform(0.5,1)         
          check = np.sum(new_encode_binary)
          if check == 0:
              while check == 0:
                  position = np.random.randint(0,len(new_encode))
                  new_encode_binary[position]= 1
                  new_encode[position] = np.random.uniform(0.5,1)
                  check = np.sum(new_encode_binary)

  #feature index extraction
  for i in range(len(new_encode)):
      if new_encode[i] >= 0.5:
          feature_subset.append(i)
                       
  return new_encode, new_encode_binary, feature_subset

def CSA_Algorithm(encode_list, encode_binary_list, feature_subset_list, cuckoo_result_list, gbest, gbest_binary, gbest_feature_subset, gbest_result):
    #CSA Algorithm
    seed=7
    np.random.seed(seed)
    iteration=5
    first_alpha=0.01
    alpha0=first_alpha
    Pa=0.25
    new_nests=int(Pa*20)
    value=math.inf
    Fnew_result=[-value,-value]
    
    for i in range(iteration):
        print('Generate new nest')
        random_encode_i=encode_list[np.random.choice(len(encode_list),replace=False)]
        new_encode, new_encode_binary, new_feature_subset=update_nest(alpha0, random_encode_i, encode_list)
        new_encode, new_encode_binary, new_feature_subset, new_cuckoo_result=fitness_of_new_nest(new_encode, new_encode_binary, new_feature_subset)
    
        print('Choose randomly nest')
        random_encode_j=encode_list[np.random.choice(len(encode_list),replace=False)]
        print('encode_j', random_encode_j)
        print('encode_i', random_encode_i)
        while (random_encode_j == random_encode_i).all():
            random_encode_j=encode_list[np.random.choice(len(encode_list),replace=False)]
        for i in range(len(encode_list)):
            equality=np.array_equal(random_encode_j,encode_list[i])
            print('equality', equality)
            #if (encode_list[i] == random_encode_j).all():
            if equality == True:
                random_encode_binary_j=encode_binary_list[i]
                random_feature_subset_j=feature_subset_list[i]
                random_cuckoo_result_list_j=cuckoo_result_list[i]
    
        print('Replace current nest')
        if new_cuckoo_result[1]>random_cuckoo_result_list_j[1]:
            #if new_cuckoo_result[1]>random_cuckoo_result_list_j[1]:
            for j in range(len(encode_list)):
                equality=np.array_equal(random_encode_j,encode_list[j])
                print('equality', equality)
                if equality == True:
                    cuckoo_result_list[j]=new_cuckoo_result
                    encode_list[j]=new_encode
                    encode_binary_list[j]=new_encode_binary
                    feature_subset_list[j]=new_feature_subset
        
        print('result', cuckoo_result_list)
        print('encode', encode_list)
        print('encode_binary', encode_binary_list)
        print('feature_subset', feature_subset_list)
    
        print('sort cuckoo nest')
        ab=np.column_stack((cuckoo_result_list, encode_list, encode_binary_list, feature_subset_list))
        print('ab', ab)
        sort=ab[np.argsort(ab[:,1])]
        print('sort', sort)
        cuckoo_result_list=sort[:, [0, 1]]
        encode_list=sort[:,2:8]
        encode_binary_list=sort[:,8:14]
        feature_subset_list=sort[:,14:]
    
        print('delete cuckoo nest according Pa')
        cuckoo_result_list=np.delete(cuckoo_result_list,[15,16,17,18,19],axis=0) 
        encode_list=np.delete(encode_list,[15,16,17,18,19],axis=0)
        encode_binary_list=np.delete(encode_binary_list, [15,16,17,18,19],axis=0)
        feature_subset_list=np.delete(feature_subset_list, [15,16,17,18,19],axis=0)
        print('size', np.array(feature_subset_list).shape)
        #feature_subset_list=np.reshape(feature_subset_list, (15,))
        #old_feature_subset_list=[]
        #for i in range(len(feature_subset_list)):
            #feature_subset_list[i]=np.reshape(feature_subset_list[i], (-1,))
            #print('try', feature_subset_list[i])
            #old_feature_subset_list.append(feature_subset_list[i])
        #a=[]
        #for i in range(len(feature_subset)):
            #feature_subset[i]=np.reshape(feature_subset[i], (-1,))
            #a.append(feature_subset[i])
            #np.array(a).shape
        
        print('generate new nest according Pa')
        new_encode_Pa_list=[]
        new_encode_binary_Pa_list=[]
        new_feature_subset_Pa_list=[]
        new_cuckoo_result_Pa_list=[]
        for i in range(new_nests):
            print('new encode for Pa',i)
            new_encode_Pa, new_encode_binary_Pa, new_feature_subset_Pa=fraction_Pa_nest(Pa, alpha0, random_encode_i, encode_list)
            new_encode_Pa, new_encode_binary_Pa, new_feature_subset_Pa, new_cuckoo_result_Pa=fitness_of_new_nest(new_encode_Pa, new_encode_binary_Pa, new_feature_subset_Pa)
            new_encode_Pa_list.append(new_encode_Pa)
            new_encode_binary_Pa_list.append(new_encode_binary_Pa)
            new_feature_subset_Pa_list.append(new_feature_subset_Pa)
            new_cuckoo_result_Pa_list.append(new_cuckoo_result_Pa)
    
        print('add new nests')
        cuckoo_result_list=np.concatenate((cuckoo_result_list,new_cuckoo_result_Pa_list), axis=0)
        encode_list=np.concatenate((encode_list, new_encode_Pa_list), axis=0)
        encode_binary_list=np.concatenate((encode_binary_list, new_encode_binary_Pa_list), axis=0)
        print('feature_subset', feature_subset_list)
        print('new_feature_subset', new_feature_subset_Pa_list)
        #print('old_feature_subset', old_feature_subset_list)
        #old_feature_subset_list=np.reshape(old_feature_subset_list,(15,))
        #print('size', np.array(old_feature_subset_list).shape)
        print('size', np.array(feature_subset_list).shape)
        print('size', np.array(new_feature_subset_Pa_list).shape)
        #new_feature_subset_Pa_list=np.reshape(new_feature_subset_Pa_list, (5,))
        #print('size', np.array(new_feature_subset_Pa_list).shape)
        feature_subset_collector=[]
        for i in range(len(feature_subset_list)):
            feature_subset_collector.append(feature_subset_list[i])
        for i in range(len(new_feature_subset_Pa_list)):
            feature_subset_collector.append(new_feature_subset_Pa_list[i])
        feature_subset_list=feature_subset_collector
        #feature_subset_list=np.concatenate(feature_subset_list, axis=0)
        #feature_subset_list=np.reshape(feature_subset_list, -1)
        #new_feature_subset_Pa_list=np.reshape(new_feature_subset_Pa_list, (5,1))
        #feature_subset_list=np.row_stack([feature_subset_list, new_feature_subset_Pa_list])
        print('try', feature_subset_list)
        
        print('finding Fnew')
        for i in range(len(cuckoo_result_list)):
            if cuckoo_result_list[i][1]>Fnew_result[1]:
                #if cuckoo_result_list[i][1]>Fnew_result[1]:
                Fnew_result=cuckoo_result_list[i]
                Fnew_encode=encode_list[i]
                Fnew_encode_binary=encode_binary_list[i]
                Fnew_feature_subset=feature_subset_list[i]
                    
        if Fnew_result[1]>gbest_result[1]:
            #if Fnew_result[1]>gbest_result[1]:
            gbest_result=Fnew_result
            gbest=Fnew_encode
            gbest_binary=Fnew_encode_binary
            gbest_feature_subset=Fnew_feature_subset
    
    return gbest, gbest_binary, gbest_feature_subset, gbest_result, seed, Pa, first_alpha

#EXECUTE
encode_list, encode_binary_list, feature_subset_list= initial_sol(20, 6)
encode_list, encode_binary_list, feature_subset_list, cuckoo_result_list = particle_best(encode_list, encode_binary_list, feature_subset_list)
#print('encode')
#print(np.array(encode_list).shape)
#print(encode_list)
#print('encode_binary') 
#print(np.array(encode_binary_list).shape)
#print(encode_binary_list)
#print('feature_subset') 
#print(np.array(feature_subset_list).shape)
#print(feature_subset_list)
#print('cuckoo_result')
#print(np.array(cuckoo_result_list).shape)
#print(cuckoo_result_list)
#ab=np.column_stack([cuckoo_result_list, encode_list, encode_binary_list, feature_subset_list])
#print(ab)
gbest, gbest_binary, gbest_feature_subset, gbest_result = global_best(encode_list, encode_binary_list, feature_subset_list, cuckoo_result_list)
best_encode, best_binary, best_feature_subset, best_result, seed, Pa, alpha = CSA_Algorithm(encode_list, encode_binary_list, feature_subset_list, cuckoo_result_list, gbest, gbest_binary, gbest_feature_subset, gbest_result)

print('best_encode', best_encode)
print('best_binary', best_binary)
print('best_feature_subset', best_feature_subset)
print('best_result', best_result)
elapsed = (time.clock() - start)
print("Time used:",elapsed)

f = open('monk2_vns_prof_test_lasttest_new(wz GPU_memory_setting)_iter_5_VNS_iter10_converge_10_BS_1_epochs_10_NP_20_RS_20_a_1_b_m2_g_0.1_threshold_0.5.txt', 'w')
f.write("Feature subset:\n")
f.write(str(best_feature_subset))
f.write("Accuracy:\n")
f.write(str(best_result))
f.write("Encode:\n")
f.write(str(best_encode))
f.write("Encode binary:\n")
f.write(str(best_binary))
f.write("Time used:\n")
f.write(str(elapsed))
f.write("Seed:\n")
f.write(str(seed))
f.write("Pa:\n")
f.write(str(Pa))
f.write("alpha:\n")
f.write(str(alpha))
f.close()
    