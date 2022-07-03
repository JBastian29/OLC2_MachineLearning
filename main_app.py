from cmath import e
from lib2to3.refactor import get_all_fix_names
from pyexpat import features
from pyrsistent import s
import streamlit as st
import pandas as pd
from st_btn_select import st_btn_select
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB


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
    optionA = " "
    optionH = " "
    
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
            st.write(datacsv[optionA])
        with col2:
            st.write('Y variable')
            # Y = Y.rename({0: 'Y'})
            st.write(Y)

        st.markdown('**1.3. Results details**:')    

        st.write("**Error medio**: ")
        st.info(mean_squared_error(Y, Y_pred, squared=True))
        st.write("**Coeficiente**: ")
        st.info(linear_regression.coef_[0])
        st.write("**R2**: ")
        st.info(r2_score(Y, Y_pred))


        with st.sidebar.header(''):
            title = st.text_input('Type the prediction data')

        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.savefig("graficaLineal.png")
        st.write("**Grafica**: ")

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
        st.write("**Funcion de tendencia**: ")
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
            st.write("**Prediction**: ")
            Y_new = linear_regression.predict([[int(titleInt)]])[0]
            st.info(Y_new)
            
        else:
            st.info('Awaiting for prediction value')

    else:
        st.info('Awaiting for X and Y axis')

def polinomial(datacsv):
    encabezados = " "
    optionH = " "
    X = []
    conta1=0
    conta2=0
    posiX=0
    posiY=0
    
    
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
        # for enca in encabezados:
        #     if(enca == optionA):
        #         posiX=conta1
        #     conta1+=1
        
        # for enca in encabezados:
        #     if(enca == optionH):
        #         posiY=conta2
        #     conta2+=1

        # conta1=0
        # conta2=0

        X = np.asarray(datacsv[optionA]).reshape(-1, 1)
        Y = datacsv[optionH]

        st.markdown('**1.2. Variable details**:')

        col1,col2 = st.columns(2)
        with col1:
            st.write('X variable')
            st.write(datacsv[optionA])
        with col2:
            st.write('Y variable')
            st.write(Y)


        

        with st.sidebar.header(''):
            grado = st.text_input('Type the degree')
        
        if grado != "":
            igrado = int(grado)
            poly = PolynomialFeatures(degree=igrado)
            Xp = poly.fit_transform(X)
            linear_regressionP = LinearRegression()
            linear_regressionP.fit(Xp,Y)


            Y_pred = linear_regressionP.predict(Xp);


            st.markdown('**1.3. Results details**:')

            st.write('**Y_pred**: ')
            st.info(Y_pred)
            st.write("**Error medio:** ")
            st.info(mean_squared_error(Y, Y_pred, squared=False))
            st.write("**R2**: ")
            st.info(r2_score(Y, Y_pred))

            with st.sidebar.header(''):
                pre = st.text_input('Type the prediction data')


            plt.scatter(X, Y)
            if(igrado > 1):
                Xgrid = np.arange(min(X),max(X),0.1)
                Xgrid = Xgrid.reshape(len(Xgrid),1)
                plt.plot(Xgrid, linear_regressionP.predict(poly.fit_transform(Xgrid)),color='red')
            if(igrado == 1):
                linear_regression = LinearRegression()
                linear_regression.fit(X,Y)
                plt.plot(X, Y_pred, color='red')


            plt.xlabel(optionA)
            plt.ylabel(optionH)
            plt.savefig("graficaPoli.png")
            st.write("**Grafica**: ")
            colu1, colu2, colu3 = st.columns([1,4,1])
            with colu1:
                st.write("")
            with colu2:
                st.image("./graficaPoli.png")
            with colu3:
                st.write("")

            if pre != "":
                titleInt = int(pre)
                st.write("**Prediction**: ")
                Y_new = linear_regressionP.predict(poly.fit_transform([[int(titleInt)]]))[0]
                st.info(Y_new)
            else:
                st.info('Awaiting for prediction value')

        else:
            st.info('Awaiting for degree value')
    else:
        st.info('Awaiting for X and Y axis')

def gaussiano(datacsv):
    features = []
    y=[]

    for d in datacsv:
        featuresT = tuple(datacsv[d])
        features.append(featuresT)
    y = list(features.pop())

    st.write('**Tuplas utilizadas**: ')
    st.info(features)

    st.write('**Eje Y**: ')
    st.info(y)

    model = GaussianNB();
    model.fit(features, y);

    with st.sidebar.header('3. Insert the prediction data: '):
        pred = st.sidebar.text_input('Prediction data: ')


    predL = list()
    for pr in pred:
        if pr != ",":
            predL.append(int(pr))
    predict = model.predict([predL])

    st.markdown('**Prediction result**: ')
    st.info(predict[0])

    

    





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
        if(option == "Clasificador Gaussiano"):
            gaussiano(data)

    
else:
    st.info('Awaiting for CSV file')