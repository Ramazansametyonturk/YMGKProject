# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:01:28 2020

@author: RamazanSamet
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv('C:/Users/samet yöntürk/Desktop/model_dataset.csv')

#Sınıf sayısı ve etiketlerin belirlenmesi
label_encoder = LabelEncoder().fit(veri.Class)
labes = label_encoder.transform(veri.Class)
classes = list(label_encoder.classes_)

#Girdi ve çıktı verilerinin hazırlanması
x = veri.drop(["Class"],axis=1)
y=labes
nb_features = 1
nb_classes = len(classes)

#Verilerin standartlaşması
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)

#Eğitim ve test verilerinin hazırlamanması
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test =train_test_split(X,y,test_size = 0.3)

# çıktı değerlerinin kategorileştirmesi
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#ÈDerin öğrenme yapabilmesi için 3 boyutlu hale getirmeliyiz
X_train = np.array(X_train).reshape(175,1,1)
X_test = np.array(X_test).reshape(76,1,1)



# modelinin oluşturulması
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Flatten, BatchNormalization

model = Sequential()

model.add(LSTM(512, input_shape=(nb_features,1)))

model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(2048, activation = "relu"))
model.add(Dense(1024, activation = "relu"))
model.add(Dense(nb_classes, activation ="softmax"))
model.summary()

#♦Modelin derlenmesi
from tensorflow.keras.optimizers import SGD
opt = SGD(lr=1e-3, decay = 1e-5,momentum = 0.9,nesterov =True)
model.compile (loss="binary_crossentropy", optimizer=opt, metrics =["accuracy"])

#modeliin Eğitilmesi
score = model.fit(X_train,y_train,epochs=50, validation_data=(X_test,y_test))

#Model tahmini
predict = np.array(X_test)
predict1=model.predict(predict)


#Gerekli bilgilerin verilmesi
print("Ortalama Başarım",np.mean(model.history.history["val_acc"]))
print("Ortalama Kayıp",np.mean(model.history.history["val_loss"]))



#Sonuçları grafiğer dökme
import matplotlib.pyplot as plt
plt.plot(model.history.history["acc"])
plt.plot(model.history.history["val_acc"])
plt.title("Model Başarımı")
plt.xlabel("Epok")
plt.ylabel("Başarım")
plt.legend(["Eğitim","Test"], loc="upper_left")
plt.show()

#Kayıpların gösterilmesi
import matplotlib.pyplot as plt
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.xlabel("Epok")
plt.ylabel("Kayıp")
plt.legend(["Eğitim","Test"], loc="upper_left")
plt.show()







