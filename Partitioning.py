import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random, os
import shutil
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

df['binary_type'] =  df['level'].map(diagnosis_dict_binary.get)
df['type'] = df['level'].map(diagnosis_dict.get)
print(df.head())

label_counts = df['type'].value_counts()
plt.figure(figsize=(10, 6))
label_counts.plot(kind='barh', color='skyblue')
plt.xlabel('Count')
plt.ylabel('Label Type')
plt.title('Count of Each Label Type')
plt.grid(axis='x')
plt.show()

# Splitting data into training (60%), validation (20%), and test (20%) sets
train_intermediate, val = train_test_split(df, test_size = 0.15, stratify = df['type'])
train, test = train_test_split(train_intermediate, test_size = 0.15 / (1 - 0.15), stratify = train_intermediate['type'])

# Print the shapes of the resulting sets
print("For Training Dataset :")
print(train['type'].value_counts(), '\n')
print("For Testing Dataset :")
print(test['type'].value_counts(), '\n')
print("For Validation Dataset :")
print(val['type'].value_counts(), '\n')

base_dir = 'F:\SEM-8\IoT Domain Analyst\IoT Jcomp\Dataset'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)

src_dir = r'F:/SEM-8/IoT Domain Analyst/Dataset/train'

for index, row in train.iterrows():
    binary_diagnosis = row['binary_type']
    id_code = row['image'] + ".jpeg"
    srcfile = os.path.join(src_dir, id_code)
    dstfile = os.path.join(train_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)
    print(index)

print("Done with train dataset")

for index, row in val.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['image'] + ".jpeg"
    srcfile = os.path.join(src_dir, id_code)
    dstfile = os.path.join(val_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)
    print(index)

print("Done with var dataset")

for index, row in test.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['image'] + ".jpeg"
    srcfile = os.path.join(src_dir, id_code)
    dstfile = os.path.join(test_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)

print("Done with test dataset")