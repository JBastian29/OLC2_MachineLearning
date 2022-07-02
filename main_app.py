from cmath import e
from sqlite3 import enable_callback_tracebacks
from pyrsistent import s
import streamlit as st
import pandas as pd
from st_btn_select import st_btn_select
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#---------------------------------#
#Global variables


#---------------------------------#



#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='OLC2 Machine Learning App',
    layout='wide')

#---------------------------------#



#---------------------------------#
st.write("""
# OLC2 Machine Learning App
""")

#---------------------------------#



#---------------------------------#
# Sidebar - Collects user input features into dataframeº
with st.sidebar.header('1. Upload your .CSV'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    

#---------------------------------#




#---------------------------------#
# Displays the dataset
def lineal(datacsv):
    encabezados = " "
    optionH = " "
    X = []
    
    for d in datacsv:
        encabezados = encabezados.strip() + d +','
    encabezados = encabezados[:-1]
    encabezados = encabezados.split(',')
    encabezados.insert(0," ")

    with st.sidebar.header('3. Choose the X axis to evaluate'):
            optionA = st.sidebar.selectbox(
            'Choose X:',
            (encabezados))

    with st.sidebar.header('4. Choose the Y axis to evaluate'):
            optionH = st.sidebar.selectbox(
            'Choose Y:',
            (encabezados))


    if optionH != " " and optionA != " ":
        X = np.asarray(datacsv[optionA]).reshape(-1, 1)
        Y = datacsv[optionH]
        linear_regression = LinearRegression()
        linear_regression.fit(X, Y)
        Y_pred = linear_regression.predict(X)

        st.markdown('**1.2. Variable details**:')

        col1,col2 = st.columns(2)
        with col1:
            st.write('X variable')
            st.write(X)
        with col2:
            st.write('Y variable')
            # Y = Y.rename({0: 'Y'})
            st.write(Y_pred)

        st.write("Error medio: ")
        st.info(mean_squared_error(Y, Y_pred, squared=True))
        st.write("Coef: ")
        st.info(linear_regression.coef_[0])
        st.write("R2: ")
        st.info(r2_score(Y, Y_pred))


        with st.sidebar.header(''):
            title = st.text_input('Type the prediction data')

        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.savefig("graficaLineal.png")
        st.write("Graficar puntos: ")

        colu1, colu2, colu3 = st.columns([1,4,1])
        with colu1:
            st.write("")
        with colu2:
            st.image("./graficaLineal.png")
        with colu3:
            st.write("")

        coe = linear_regression.coef_[0]
        inter=linear_regression.intercept_
        coe2="{0:.4f}".format(coe)
        inter2="{0:.4f}".format(inter)
        st.write("Funcion de tendencia: ")
        if float(coe2)>0:
            st.latex(f'''
            y = {str(inter2)} + {str(coe2)} x
            ''')
        else:
            st.latex(f'''
            y = {str(inter2)} {str(coe2)} x
            ''')

        
        if title != "":
            titleInt = int(title)
            st.write("Prediction: ")
            Y_new = linear_regression.predict([[int(titleInt)]])[0]
            st.info(Y_new)
            
        else:
            st.info('Awaiting for prediction value')

    else:
        st.info('Awaiting for X and Y axis')




def polinomial(datacsv):
    pass





if uploaded_file is not None:
    st.subheader('1. Dataset')
    data = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Data set overview**')
    st.write(data)


    with st.sidebar.header('2. What do you want to do?'):
     option = st.sidebar.selectbox(
      'Choose an algorithm:',
     ('','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano','Clasificador de árboles de decisión','Redes neuronales'))

    if(option != ""):
        if(option == "Regresión lineal"):
            lineal(data)
        if(option == "Regresión polinomial"):
            polinomial(data)

        # with st.sidebar.header('5. Operations'):
        #     algoSelected = st.sidebar.selectbox(
        #     'Choose an operation:',
        #     ('','Graficar puntos', ' Definir función de tendencia', 'Realizar predicción de la tendencia','Clasificar'))
    
else:
    st.info('Awaiting for CSV file')