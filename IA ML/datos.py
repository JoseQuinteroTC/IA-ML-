

import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial.distance import cosine
from unidecode import unidecode
from sklearn.tree import plot_tree

# Gráficos
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from wordcloud import WordCloud

# Preprocesado y modelado
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


# Configuración warnings
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import warnings


df = pd.read_csv("train_new.txt", 
                      delimiter = '|') 
  
# storing this dataframe in a csv file 
df.to_csv('hola.csv',  
                  index = None) 



print(df.head())

# Contar el total de filas
total_filas = df.shape[0]

print("\nTotal de filas:", total_filas)

# Establecer el tamaño de la figura
plt.figure(figsize=(8, 5))

# Crear un countplot con las categorías "en" y "nl" de diferentes colores
sns.countplot(data=df, x='idioma', palette={'en': 'blue', 'nl': 'black'})

# Configurar los ejes y etiquetas
plt.xlabel('Idioma')
plt.ylabel('Cantidad')

# Mostrar el gráfico
plt.show()

def limpiar_tokenizar(texto):
    # Crear instancia del lematizador
    lemmatizer = WordNetLemmatizer()
    
    nuevo_texto = texto.lower()
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Eliminación de diéresis
    nuevo_texto = unidecode(nuevo_texto)
    # Eliminación de fracciones
    nuevo_texto = re.sub("\d+/\d+", ' ', nuevo_texto)
    # Eliminación de fechas
    nuevo_texto = re.sub("\d{1,2}/\d{1,2}/\d{2,4}", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep=' ')
    # Lematización
    nuevo_texto = [lemmatizer.lemmatize(token) for token in nuevo_texto]

    
    return nuevo_texto

df['texto_tokenizado'] = df['texto'].apply(lambda x: limpiar_tokenizar(x))
df[['idioma','texto','texto_tokenizado']].to_csv('hola.csv', index=False)


texto_tidy =df.explode(column='texto_tokenizado')
texto_tidy= texto_tidy.drop(columns='texto')
texto_tidy= texto_tidy.rename(columns={'texto_tokenizado':'token'})
texto_tidy.to_csv('texto_tidy.csv', index=False)

print(texto_tidy.head(10))



print("\nTotal de filas:", texto_tidy.shape[0])


# Import the required libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Convert float items to strings in the palabras list
palabras_por_categoria = texto_tidy.groupby('idioma')['token'].apply(lambda x: [str(word) for word in x])

# Create an instance of WordCloud for each category and generate the word cloud
wordclouds = {}
for categoria, palabras in palabras_por_categoria.items():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(palabras))
    wordclouds[categoria] = wordcloud

# Display the word clouds for each category
plt.figure(figsize=(12, 6))
for i, (categoria, wordcloud) in enumerate(wordclouds.items()):
    plt.subplot(1, 2, i+1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(categoria)
    plt.axis('off')
plt.tight_layout()
plt.show()


"""
#Cargando datos
diabetes = pd.read_excel("./datos_excel.xlsx")
#Informacion de los datos


print("\n===================================\n Resumen de los datos \n")
print(diabetes.info(),"\n===================================")

print("\n===================================\n Cantidad de datos nulos por cada columna \n")
print(diabetes.isnull().sum(),"\n===================================")

print("\n===================================\n Resumen estadistico \n")
print(diabetes.describe(),"\n===================================")



#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(15,15))
sns.countplot(x='idioma',data=diabetes)
plt.title("Outcome Count")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(diabetes['texto'], diabetes['idioma'], train_size=0.2)

vectorizer = CountVectorizer()
vector = vectorizer.fit_transform(X_train)

df = pd.DataFrame(vector, columns =['x'])
print(df.head())
df.to_excel('x.xlsx')
"""

















