import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



import warnings

#Cargando datos
diabetes = pd.read_excel("./datos_excel2.xlsx")
#Informacion de los datos
print(diabetes.info())
print("================")
print(diabetes.isnull().sum())

#Resumen de estadísticos
diabetes.describe()
print("================")
print(diabetes.describe())


#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot(x='Idioma',data=diabetes)
plt.title("Outcome Count")
plt.show()
