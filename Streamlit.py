import os
import glob
import json
import numpy as np

import tensorflow.keras as k
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as k
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import streamlit as st

# 设置混合精度策略来加速模型训练
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Streamlit应用程序标题
st.header("木薯叶病检测")

# 图像大小设置
IMG_SIZE = 512
size = (IMG_SIZE, IMG_SIZE)

# 加载预训练的最佳模型
best_model = k.models.load_model('models/Cassava_best_model_effnetb4.h5', compile=False)
print(best_model)

# 测试图像目录
TEST_DIR = 'test_images/'
test_images = []
test_images.append(st.file_uploader('文件上传器'))

predictions = []

# 创建一个按钮，当点击按钮时开始预测
button = st.button("开始预测")

if button:
    with st.spinner(text="请稍等..."):
        for image in test_images:
            if image is not None:
                img = Image.open(image)
                img = img.resize(size)
                print(img)
                img = np.expand_dims(img, axis=0)
                print(img.shape)
                predictions.extend(best_model.predict(img).argmax(axis=1))
                values = best_model.predict(img)[0]

                # 显示预测概率值
                st.text('概率值:' + str(values))
                data = pd.DataFrame({'values': values})
                st.bar_chart(data)

                # 显示类别标签
                st.text("""
                0 : 木薯细菌枯萎病 (CBB)
                1 : 木薯花叶病 (CMD)
                2 : 木薯褐条病 (CBSD)
                3 : 木薯绿斑病 (CGM)
                4 : 健康
                """)

                # 显示预测结果
                st.success("预测结果如下")
