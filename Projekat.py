# Databricks notebook source
# MAGIC %scala
# MAGIC val containerName="blobpodaci"
# MAGIC val storageAccountName="blobprojekat"
# MAGIC val sas="?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacuptfx&se=2022-06-29T22:06:36Z&st=2021-08-10T14:06:36Z&spr=https,http&sig=w5a0xPqs3ofxQtAQJpSQheERjjNnx%2FmmEOKLqEu8x3s%3D"
# MAGIC val url= "wasbs://"+containerName+"@"+storageAccountName+".blob.core.windows.net/"
# MAGIC val config="fs.azure.sas."+containerName+"."+storageAccountName+".blob.core.windows.net"

# COMMAND ----------

# MAGIC %scala
# MAGIC dbutils.fs.mount(
# MAGIC   source=url,
# MAGIC   mountPoint="/mnt/projekat_data/aktivnosti_data",
# MAGIC   extraConfigs=Map(config->sas)
# MAGIC )

# COMMAND ----------



df = spark.read.option("header", "false").csv("/mnt/projekat_data/aktivnosti_data/Akcelometar_podaci.txt")



# COMMAND ----------

display(df)

# COMMAND ----------

columns=["UserID","Aktivnost","Time_ns","X_osa","Y_osa","Z_osa"]
df=df.toDF(*columns)
df.show()


# COMMAND ----------

df_P=df.toPandas()
df_P.isnull().sum()

# COMMAND ----------


df.write.format("delta").mode("overwrite").save("/projekat_data/aktivnosti_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC create database projekat_data;
# MAGIC CREATE TABLE projekat_data.aktivnosti_data
# MAGIC USING DELTA LOCATION '/projekat_data/aktivnosti_data'

# COMMAND ----------

# MAGIC %sql select * from projekat_data.aktivnosti_data where Aktivnost='Jogging' and UserID='33'

# COMMAND ----------

spark.table("projekat_data.aktivnosti_data").printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE projekat_data.aktivnosti_trans_data
# MAGIC USING DELTA LOCATION '/projekat_data/aktivnosti_trans_data'
# MAGIC AS (
# MAGIC   SELECT cast(UserID as INT) as UserID,Aktivnost,Time_ns,cast(X_osa as float) as X_osa,cast(Y_osa as float) as Y_osa,cast((TRIM(';' from Z_osa)) as float) as Z_osa
# MAGIC   FROM projekat_data.aktivnosti_data
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select * 
# MAGIC from projekat_data.aktivnosti_trans_data
# MAGIC where Aktivnost='Jogging' and UserID='33'

# COMMAND ----------

data=spark.table("projekat_data.aktivnosti_trans_data")
data.printSchema()


# COMMAND ----------

print((data.count(), len(data.columns)))

# COMMAND ----------

from pyspark.sql.functions import count,when,isnan,col
data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()


# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from projekat_data.aktivnosti_trans_data
# MAGIC where Z_osa is null

# COMMAND ----------

# MAGIC %sql select * from projekat_data.aktivnosti_trans_data where UserID=30 and Aktivnost='Standing'

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE projekat_data.aktivnosti_trans_data
# MAGIC set  Z_osa = (select avg(Z_osa)
# MAGIC                 from projekat_data.aktivnosti_trans_data
# MAGIC                 where Z_osa is not NUll and UserID=30 and Aktivnost='Standing'
# MAGIC                 group by UserID,Aktivnost)
# MAGIC where Z_osa is null and UserID=30 and Aktivnost='Standing'

# COMMAND ----------

# MAGIC %sql select * from projekat_data.aktivnosti_trans_data where UserID=11 and Aktivnost='Walking'

# COMMAND ----------

pd=spark.sql("select Z_osa from projekat_data.aktivnosti_trans_data where UserID=11 and Aktivnost='Walking'").toPandas()
med=pd.median().iloc[0]

# COMMAND ----------

med

# COMMAND ----------

# MAGIC %sql
# MAGIC update projekat_data.aktivnosti_trans_data
# MAGIC set Z_osa = -0.040861044
# MAGIC where UserID=11 and Aktivnost='Walking' and Z_osa is null

# COMMAND ----------

print((data.count(), len(data.columns)))

# COMMAND ----------

# MAGIC %sql 
# MAGIC delete from  projekat_data.aktivnosti_trans_data
# MAGIC where Z_osa is null

# COMMAND ----------

data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()

# COMMAND ----------

# MAGIC %sql 
# MAGIC select Aktivnost,count(*) as brAktivnosti
# MAGIC from projekat_data.aktivnosti_trans_data
# MAGIC group by Aktivnost
# MAGIC order by brAktivnosti desc

# COMMAND ----------

pip install -U imbalanced-learn

# COMMAND ----------

data=data.drop("UserID","Time_ns")
data.show()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Aktivnost", outputCol="Label") 
data = indexer.fit(data).transform(data)
data.show()

# COMMAND ----------

# MAGIC %sql 
# MAGIC select distinct(Aktivnost), Label 
# MAGIC from projekat_data.aktivnosti_extraxt_data 

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(Label) as Labela, Label
# MAGIC from  projekat_data.aktivnosti_extraxt_data
# MAGIC group by Label
# MAGIC order by Labela desc

# COMMAND ----------

data.write.format("delta").mode("overwrite").save("/projekat_data/aktivnosti_extraxt_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE projekat_data.aktivnosti_extraxt_data
# MAGIC USING DELTA LOCATION '/projekat_data/aktivnosti_extraxt_data'

# COMMAND ----------

X= spark.table("projekat_data.aktivnosti_extraxt_data").drop("Aktivnost","Label")
Y=spark.table("projekat_data.aktivnosti_extraxt_data").select("Label")

# COMMAND ----------

from imblearn.over_sampling import SMOTE
smote = SMOTE('minority')
x=X.toPandas()
y=Y.toPandas()
x_sm,y_sm = smote.fit_resample(x,y)

# COMMAND ----------

x_sm.shape,y_sm.shape

# COMMAND ----------

y_sm.value_counts()

# COMMAND ----------

# MAGIC %sql 
# MAGIC select Aktivnost,Label,count(*) as brAktivnosti
# MAGIC from projekat_data.aktivnosti_extraxt_data
# MAGIC group by Aktivnost,Label
# MAGIC order by brAktivnosti desc

# COMMAND ----------

# MAGIC %sql 
# MAGIC select Label,count(*) as brAktivnosti
# MAGIC from projekat_data.smote_y_data
# MAGIC group by Label
# MAGIC order by brAktivnosti desc

# COMMAND ----------

# MAGIC %sql select *
# MAGIC from projekat_data.smote_y_data

# COMMAND ----------

df_x_sm=spark.createDataFrame(x_sm)
df_y_sm=spark.createDataFrame(y_sm)

# COMMAND ----------

df_x_sm.write.format("delta").mode("overwrite").save("/projekat_data/smote_x_data");
df_y_sm.write.format("delta").mode("overwrite").save("/projekat_data/smote_y_data");

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE projekat_data.smote_x_data
# MAGIC USING DELTA LOCATION '/projekat_data/smote_x_data';
# MAGIC 
# MAGIC CREATE TABLE projekat_data.smote_y_data
# MAGIC USING DELTA LOCATION '/projekat_data/smote_y_data';

# COMMAND ----------

X_sm=spark.table("projekat_data.smote_x_data").toPandas()
Y_sm=spark.table("projekat_data.smote_y_data").toPandas()

# COMMAND ----------

Y_sm.value_counts()

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from projekat_data.smote_x_data

# COMMAND ----------

X_sm.head()

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from projekat_data.smote_y_data

# COMMAND ----------

Y_sm.head()

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
import pandas as pd
scaler=StandardScaler()
X=scaler.fit_transform(X_sm)
sc_x=pd.DataFrame(data=X, columns = ['x_osa','y_osa','z_osa'])



# COMMAND ----------

sc_x.head()

# COMMAND ----------

sc_x_sp=spark.createDataFrame(sc_x)
sc_x_sp.write.format("delta").mode("overwrite").save("/projekat_data/std_scaling_x_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE or replace TABLE projekat_data.std_scaling_x_data
# MAGIC USING DELTA LOCATION '/projekat_data/std_scaling_x_data'

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from projekat_data.std_scaling_x_data

# COMMAND ----------

sc_x['Label']=Y_sm


# COMMAND ----------

sc_x=spark.table("projekat_data.std_scaling_x_data").toPandas()
Y_sm=spark.table("projekat_data.smote_y_data").toPandas()

# COMMAND ----------

sc_x.head()

# COMMAND ----------

import scipy.stats as stats
import numpy as np
frame_size=80
skip=80

# COMMAND ----------

def create_frame (df, frame_size, skip):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, skip):
        x = df['x_osa'].values[i: i + frame_size]
        y = df['y_osa'].values[i: i + frame_size]
        z = df['z_osa'].values[i: i + frame_size]
        label = stats.mode(df['Label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

   
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

  

# COMMAND ----------

X,y= create_frame(sc_x,frame_size,skip)
X.shape,y.shape

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# COMMAND ----------

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# COMMAND ----------

X_train = X_train.reshape(14741,8,-1,3)
X_test = X_test.reshape(3686,8,-1,3)

# COMMAND ----------

X_train.shape,X_test.shape

# COMMAND ----------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten

# COMMAND ----------

X_train[0].shape


# COMMAND ----------

model = Sequential()
model.add(Conv2D(16, 2, activation = 'relu', input_shape = X_train[0].shape, name="convl1"))
model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))

model.add(Conv2D(32, 2, activation='relu',name="convl2"))
model.add(MaxPool2D(pool_size=2,strides=(1,1)))
          
model.add(Flatten())
          
model.add(Dense(64, activation = 'relu',name="fully_con1"))


model.add(Dense(32, activation = 'relu',name="fully_con2"))

          
model.add(Dense(6, activation='softmax',name="prediction"))

# COMMAND ----------

model.summary()

# COMMAND ----------

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
model.compile(optimizer=Adam(learning_rate=0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 15, validation_data= (X_test, y_test), verbose=2)

# COMMAND ----------

type(history)

# COMMAND ----------

import pandas as pd

# COMMAND ----------

import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(10, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2) 
plt.show()

# COMMAND ----------

pip install mlxtend

# COMMAND ----------

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
y_pred

# COMMAND ----------

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
y_pred

# COMMAND ----------

mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat,figsize=(10,10))

# COMMAND ----------

# MAGIC %sql 
# MAGIC select distinct(Aktivnost), Label 
# MAGIC from projekat_data.aktivnosti_extraxt_data 

# COMMAND ----------

# MAGIC %sql select * from projekat_data.aktivnosti_data

# COMMAND ----------

df_pd=spark.table("projekat_data.aktivnosti_data").toPandas()
df_pd.head()
