from cmath import e
from lib2to3.refactor import get_all_fix_names
from pyexpat import features
from pyparsing import empty
from pyrsistent import s
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from streamlit_option_menu import option_menu
import pathlib
from sklearn.neural_network import MLPClassifier




st.set_page_config(page_title='OLC2 Machine Learning App',
    layout='wide')


selected3 = option_menu(None, ["Home","R. lineal", "R. polinomial",  "C. Gaussiano", "Árb. decisión", "Redes neuro."], 
    icons=['house'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#eee"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "color": "black" ,"margin":"0px", "--hover-color": "#00816D" , "font-weight":"bold"},
        "nav-link-selected": {"background-color": "#01D063"},
    }
    )


st.write("""
# OLC2 Machine Learning App
""")



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

    with st.sidebar.header('2. Choose the X axis to evaluate'):
            optionA = st.sidebar.selectbox(
            'Choose X:',
            (encabezados))

    with st.sidebar.header('3. Choose the Y axis to evaluate'):
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
            st.write(pd.DataFrame(datacsv[optionA]))
        with col2:
            st.write('Y variable')
            st.write(pd.DataFrame(Y))

        st.markdown('**1.3. Results details**:')    

        st.write("**Error medio**: ")
        st.info(mean_squared_error(Y, Y_pred, squared=True))
        st.write("**Coeficiente**: ")
        st.info(linear_regression.coef_[0])
        st.write("**R2**: ")
        st.info(r2_score(Y, Y_pred))


        with st.sidebar.header(''):
            title = st.text_input('Type the prediction data')

        figura = plt.figure()
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.xlabel(optionA)
        plt.ylabel(optionH)
        st.write("**Grafica**: ")
        colu1, colu2, colu3 = st.columns([1,4,1])
        with colu1:
            st.write("")
        with colu2:
            st.pyplot(figura)
        with colu3:
            st.write("")

        # chart_data = pd.DataFrame(np.random.randn(20,27),columns=[encabezados])
        # st.line_chart(chart_data)
        
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

    with st.sidebar.header('2. Choose the X axis to evaluate'):
            optionA = st.sidebar.selectbox(
            'Choose X:',
            (encabezados))

    with st.sidebar.header('3. Choose the Y axis to evaluate'):
            optionH = st.sidebar.selectbox(
            'Choose Y:',
            (encabezados))


    if optionH != " " and optionA != " ":


        X = np.asarray(datacsv[optionA]).reshape(-1, 1)
        Y = datacsv[optionH]

        st.markdown('**1.2. Variable details**:')

        col1,col2 = st.columns(2)
        with col1:
            st.write('X variable')
            st.write(pd.DataFrame(datacsv[optionA]))
        with col2:
            st.write('Y variable')
            st.write(pd.DataFrame(Y))


        

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

            figura = plt.figure()
            plt.scatter(X, Y)
            if(igrado > 1):
                Xgrid = np.arange(min(X),max(X),0.1)
                Xgrid = Xgrid.reshape(len(Xgrid),1)
                plt.plot(Xgrid, linear_regressionP.predict(poly.fit_transform(Xgrid)),color='red')
            if(igrado == 1):
                plt.plot(X, Y_pred, color='red')


            plt.xlabel(optionA)
            plt.ylabel(optionH)
            st.write("**Grafica**: ")
            colu1, colu2, colu3 = st.columns([1,4,1])
            with colu1:
                st.write("")
            with colu2:
                st.pyplot(figura)
            with colu3:
                st.write("")

            inter=linear_regressionP.intercept_
            inter2="{0:.4f}".format(inter)
            funcT = ""
            conta = 1
            for x in linear_regressionP.coef_:
                c = "{0:.4f}".format(x)
                print(x)
                if(conta>1):
                    if float(x) > 0:
                        funcT += "+"+str(c)+"x^"+str(conta)
                        conta += 1
                    elif float(x)<0:
                        funcT += str(c)+"x^"+str(conta)
                        conta += 1
                else:
                    if float(x) > 0:
                        funcT += "+"+str(c)+"x"
                        conta += 1
                    elif float(x)<0:
                        funcT += str(c)+"x"
                        conta += 1

            st.latex(f'''y = {str(inter2)}  {funcT} ''')



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
    encabezados = " "
    features = []
    y=[]
    featureEncoded = list()
    noEncoded = list()
    newFeatures=[]
    le = preprocessing.LabelEncoder()

    optionMultiSelect = st.multiselect('Select the columns to use: ',datacsv.columns)


    if(bool(optionMultiSelect)):
        for option in optionMultiSelect:
            for d in datacsv:
                if(d == option):
                    featuresT = tuple(le.fit_transform(datacsv[d]))
                    features.append(featuresT)
                    noEncoded.append(datacsv[d].astype(str))
        

        for i in range(len(features[0])):
            for fila in features:
                columna = fila[i]
                newFeatures.append(columna)
            featureEncoded.append(tuple(newFeatures))
            newFeatures=[]

        for d in datacsv:
            encabezados = encabezados.strip() + d +','
        encabezados = encabezados[:-1]
        encabezados = encabezados.split(',')
        encabezados.insert(0," ")

        ySelected = st.sidebar.selectbox(
            'Choose Class data:',
            (encabezados))
        

        if ySelected != " ":
            y = list(le.fit_transform(datacsv[ySelected]))
            st.write('**Tuplas utilizadas**: ')
            # st.write(noEncoded)
            st.write(pd.DataFrame(noEncoded))

            st.write('**Tuplas encoded**: ')
            colu1, colu2 = st.columns(2)
            with colu1:
                st.info(featureEncoded)
            with colu2:
                st.write(pd.DataFrame(featureEncoded))
            

            st.write('**Class**: ')
            st.info(y)

            model = GaussianNB();
            model.fit(featureEncoded, y);

            with st.sidebar.header('2. Insert the prediction data: '):
                pred = st.sidebar.text_input('Prediction data: ')
                pred2 = pred.strip()

            if pred != "":
                predL = list()
                for pr in pred2:
                    if pr != "," and pr != " " and pr != ", ":
                        predL.append(int(pr))
                predict = model.predict([predL])

                st.markdown('**Prediction result**: ')
                st.info(predict[0])
            else:
                st.info('Awaiting for prediction value')
        else:
            st.info('Awaiting for Class data')
    else:
        st.info('Awaiting for the columns')

def arbolG(datacsv):
    encabezados=""
    features = []
    y=[]
    featureEncoded = list()
    noEncoded = list()
    newFeatures=[]
    le = preprocessing.LabelEncoder()
    optionMultiSelect = st.multiselect('Select the columns to use: ',datacsv.columns)

    if(bool(optionMultiSelect)):
        for option in optionMultiSelect:
            for d in datacsv:
                if(d == option):
                    featuresT = tuple(le.fit_transform(datacsv[d]))
                    features.append(featuresT)
                    noEncoded.append(datacsv[d].astype(str))

        for i in range(len(features[0])):
            for fila in features:
                columna = fila[i]
                newFeatures.append(columna)
            featureEncoded.append(tuple(newFeatures))
            newFeatures=[]


        for d in datacsv:
            encabezados = encabezados.strip() + d +','
        encabezados = encabezados[:-1]
        encabezados = encabezados.split(',')
        encabezados.insert(0," ")

        ySelected = st.sidebar.selectbox(
            'Choose Class data:',
            (encabezados))
        

        if ySelected != " ":
            y = list(le.fit_transform(datacsv[ySelected]))
            st.write('**Tuplas utilizadas**: ')
            # st.write(noEncoded)
            st.write(pd.DataFrame(noEncoded))

            st.write('**Tuplas encoded**: ')
            colu1, colu2 = st.columns(2)
            with colu1:
                st.info(featureEncoded)
            with colu2:
                st.write(pd.DataFrame(featureEncoded))

            st.write('**Class**: ')
            st.info(y)

            treeClass = DecisionTreeClassifier()
            model = treeClass.fit(featureEncoded, y)

            with st.sidebar.header('2. Insert the prediction data: '):
                pred = st.sidebar.text_input('Prediction data: ')
                pred2 = pred.strip()

            if pred != "":
                predL = list()
                for pr in pred2:
                    if pr != "," and pr != " " and pr != ", ":
                        predL.append(int(pr))
                predict = model.predict([predL])

                st.markdown('**Prediction result**: ')
                st.info(predict[0])

                figura = plt.figure()
                plot_tree(model, filled=True)
                st.write("**Arbol de decisión**: ")
                colu1, colu2, colu3 = st.columns([2,10,1])
                with colu1:
                    st.write("")
                with colu2:
                    st.pyplot(figura)
                with colu3:
                    st.write("")
            else:
                st.info('Awaiting for prediction value')
        else:
            st.info('Awaiting for Class data')
    else:
        st.info('Awaiting for the columns')

def neuronales(datacsv):
    encabezados=" "
    features = []
    y=[]
    featureEncoded = list()
    noEncoded = list()
    newFeatures=[]
    le = preprocessing.LabelEncoder()
    optionMultiSelect = st.multiselect('Select the columns to use: ',datacsv.columns)


    if(bool(optionMultiSelect)):
        for option in optionMultiSelect:
            for d in datacsv:
                if(d == option):
                    featuresT = tuple(le.fit_transform(datacsv[d]))
                    features.append(featuresT)
                    noEncoded.append(datacsv[d].astype(str))

        for i in range(len(features[0])):
            for fila in features:
                columna = fila[i]
                newFeatures.append(columna)
            featureEncoded.append(tuple(newFeatures))
            newFeatures=[]


        for d in datacsv:
            encabezados = encabezados.strip() + d +','
        encabezados = encabezados[:-1]
        encabezados = encabezados.split(',')
        encabezados.insert(0," ")

        ySelected = st.sidebar.selectbox(
            'Choose Class data:',
            (encabezados))
        

        if ySelected != " ":
            y = list(le.fit_transform(datacsv[ySelected]))
            st.write('**Tuplas utilizadas**: ')
            st.write(pd.DataFrame(noEncoded))

            st.write('**Tuplas encoded**: ')
            colu1, colu2 = st.columns(2)
            with colu1:
                st.info(featureEncoded)
            with colu2:
                st.write(pd.DataFrame(featureEncoded))

            st.write('**Class**: ')
            st.info(y)

            model=MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, alpha=0.0001,solver='adam', random_state=21,tol=0.000000001)
            model.fit(featureEncoded, y)

            with st.sidebar.header('2. Insert the prediction data: '):
                pred = st.sidebar.text_input('Prediction data: ')
                pred2 = pred.strip()

            if pred != "":
                predL = list()
                for pr in pred2:
                    if pr != "," and pr != " " and pr != ", ":
                        predL.append(int(pr))
                predict = model.predict([predL])

                st.markdown('**Prediction result**: ')
                st.info(predict[0])
            else:
                st.info('Awaiting for prediction value')
        else:
            st.info('Awaiting for Class data')
    else:
        st.info('Awaiting for the columns')


    


with st.sidebar.header('1. Upload your File'):
    archivoCargado = st.sidebar.file_uploader("Upload your input file", type=["csv", "xls", "xlsx", "json"])



if archivoCargado is not None:
    path = pathlib.Path(archivoCargado.name)
    data = ""
    st.subheader('1. Dataset')
    if (path.suffix == ".csv"):
        data = pd.read_csv(archivoCargado)
        st.markdown('**1.1. Data set overview**')
        df = pd.DataFrame(data)
        st.dataframe(df)
    if(path.suffix == '.xls' or path.suffix == '.xlsx'):
        data = pd.read_excel(archivoCargado)
        st.markdown('**1.1. Data set overview**')
        df = pd.DataFrame(data)
        st.dataframe(df)
    if(path.suffix == '.json'):
        data = pd.read_json(archivoCargado)
        st.markdown('**1.1. Data set overview**')
        df = pd.DataFrame(data)
        st.dataframe(df)

    if(selected3 != ""):
        if(selected3 == "R. lineal"):
            lineal(data)
        if(selected3 == "R. polinomial"):
            polinomial(data)
        if(selected3 == "C. Gaussiano"):
            gaussiano(data)
        if(selected3 == "Árb. decisión"):
            arbolG(data)
        if(selected3 == "Redes neuro."):
            neuronales(data)
    
    #grid_table = AgGrid(df,fit_columns_on_grid_load= True, theme='fresh')
    # st.write(grid_table)
    #AgGrid(data)
    #st.write(len(data))


    # with st.sidebar.header('2. What do you want to do?'):
    #  option = st.sidebar.selectbox(
    #   'Choose an algorithm:',
    #  ('','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano','Árboles de decisión','Redes neuronales'))

    # if(option != ""):
    #     if(option == "Regresión lineal"):
    #         lineal(data)
    #     if(option == "Regresión polinomial"):
    #         polinomial(data)
    #     if(option == "Clasificador Gaussiano"):
    #         gaussiano(data)
    #     if(option == "Árboles de decisión"):
    #         arbolG(data)

    

    
else:
    st.info('Awaiting for your file')