import os
import glob
import random
data = 'Kitti'
data_store = 'npydata'
label_store = 'npylabel'
root =os.getcwd()

data_path = os.path.join(root,data,data_store)
label_path = os.path.join(root,data,label_store)
print(data_path,label_path)
if not os.path.exists(data_path):
    print('no path exist')
    os.makedirs(data_path)
if not os.path.exists(label_path):
    os.makedirs(label_path)
    print('no path exist')
    
    
train = open('train.txt','w')
val = open('valid.txt','w')

file_path=os.path.join(data_path,'*.npy')
file_P = glob.glob(file_path)
L=len(file_P)
random.shuffle(file_P)
train_len = int(L*0.8)
