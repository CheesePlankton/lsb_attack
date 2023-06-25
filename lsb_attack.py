#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pickle


# These two two blocks are the original code to train the model.

# In[4]:


# 数据准备
def prepare_data(directory):
    X = []  # 特征向量列表
    y = []  # 标签列表
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        for file in os.listdir(file_path):
            image_path=os.path.join(file_path,file)
            # 提取特征
            features = extract_features(image_path)
            X.append(features)
            # 标记标签
            if "stego" in filename:
                label = 1 
            elif "clean" in filename:
                label=0
            y.append(label)
    print("data prepared")
    return np.array(X), np.array(y)

# 提取特征（灰度共生矩阵）
def extract_features(image_path):
    image = Image.open(image_path).convert('L')  # 转为灰度图像
    image_array = np.array(image)
    # 计算灰度共生矩阵
    glcm = graycomatrix(image_array, distances=[1], angles=[0], symmetric=True, normed=True)
    # 提取灰度共生矩阵的统计特征（例如对比度、能量、相关性等）
    features = graycoprops(glcm, prop='contrast'), graycoprops(glcm, prop='energy'),graycoprops(glcm, prop='correlation')
    return np.hstack(features)

# 数据划分
def split_data(X, y):
    return train_test_split(X, y,  random_state=10)

# 模型训练
def train_model(X_train, y_train):
    
    print("model training...")
    model = svm.SVC(kernel='linear')  # 使用线性核的SVM模型
    model.fit(X_train, y_train)
    print("model trained")
    return model

# 模型评估
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall




# In[5]:


# 数据准备
data_directory = r'D:\download\archive\train\train'
X, y = prepare_data(data_directory)

# 数据划分
X_train, X_test, y_train, y_test = split_data(X, y)
X_train = (X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# 模型训练
model = train_model(X_train, y_train)

# 模型评估
accuracy, precision, recall = evaluate_model(model, X_test, y_test)
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# # 运行主函数
# if __name__ == "__main__":
#     main()


# This block is to load the trained model.Please run this one:

# In[ ]:


# Specify the file path of the trained SVM model
model_file = './svm_model.pkl'

# Load the SVM model
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Now you can use the loaded SVM model for prediction or other tasks


# This will detect the test.png whether it hides secret message:

# In[21]:


# Load the image you want to test
image_path = r'C:\Users\z2745\Desktop\lsb\car.png'
features = extract_features(image_path)
features = features.reshape(1, -1)

# Use the trained model to predict
prediction = model.predict(features)[0]
if prediction == 1:
    print("The image contains a secret message.Its contrast, energy and correlation are showed below:")
    print(features[0])    
else:
    print("The image does not contain a secret message.")


# This block can extract the LSB of pixels, convert the LSB to text, convert the LSB to image:

# In[3]:


from PIL import Image
import numpy as np

def Extract_LSB(image_path):
    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    # Extract LSBs
    binary_message = ""
    for y in range(height):
        for x in range(width):
            pixel = list(image.getpixel((x, y)))
            for channel in range(3):  # RGB channels
                binary_message += str(pixel[channel] & 1)  # Extract the LSB
    
    return binary_message

def extract_text(binary_message): 
    # Convert binary message to characters
    characters = []
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        characters.append(chr(int(byte, 2)))
        
    
    return ''.join(characters)

def decrypt_image(encrypted_image_path):
    # Load the encrypted image
    encrypted_image = Image.open(encrypted_image_path).convert('L')
    encrypted_data = np.array(encrypted_image)

    # Get image size
    width, height = encrypted_image.size

    # Create an empty image for decrypted pixels
    decrypted_data = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel and extract LSB
    for y in range(height):
        for x in range(width):
            pixel = encrypted_data[y][x]

            # Convert pixel value to binary
            binary_pixel = bin(pixel)[2:].zfill(8)

            # Extract the LSB and add 7 zeros
            decrypted_pixel = int(binary_pixel[-1] + '0000000', 2)

            # Assign the decrypted pixel value
            decrypted_data[y][x] = decrypted_pixel

    # Create decrypted image from decrypted pixel data
    decrypted_image = Image.fromarray(decrypted_data)

    return decrypted_image

# Usage example
encrypted_image_path = r'C:\Users\z2745\Desktop\lsb\car.png'

decrypted_image = decrypt_image(encrypted_image_path)
decrypted_image.save(r'C:\Users\z2745\Desktop\lsb\secret_image.png')

print("Encrypted image successfully decrypted and saved as 'decrypted_image.png'")


# This block can modify the LSB of each 4 pixel of image, also can set the nnum_bits:

# In[24]:


def modify_LSBs(image_path, num_bits):
    image = Image.open(image_path).convert('RGB')
    pixels = image.load()
    # Modify the LSBs of the pixels
    for i in range(0,image.width,4):
        for j in range(0,image.height,4):
            r, g, b = pixels[i, j]
            # Modify the least significant bits (LSBs)
            r = (r & ~(2 ** num_bits - 1))  # Reset the least significant bits to 0
            g = (g & ~(2 ** num_bits - 1))
            b = (b & ~(2 ** num_bits - 1))
            pixels[i, j] = (r, g, b)
    # Save the modified image
    modified_image_path = r'C:\Users\z2745\Desktop\lsb\modified_image.'+image_path.split('.')[1]
    image.save(modified_image_path)
    return modified_image_path

# Modify the LSBs if the image contains a secret message
if prediction == 1:
    modified_image_path = modify_LSBs(r'C:\Users\z2745\Desktop\lsb\car.png', num_bits=2)
    print("Modified image with disrupted secret message saved as: {}".format(modified_image_path))


# In[ ]:




