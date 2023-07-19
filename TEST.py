import streamlit as st
from streamlit_elements import elements, mui, html
from streamlit_extras.metric_cards import style_metric_cards
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder



train_data = pd.read_csv("serviceTrainData.csv")
test_data = pd.read_csv("serviceTestData.csv")


st.set_page_config(page_title="Service Webapp",layout="wide")     ### Tab name
style_metric_cards()
st.markdown("<h1 style='text-align: center; FONT COLOR=#FFFF99;'>Service Prediction - By Jasmine</h1>", unsafe_allow_html=True)
#st.markdown("<H1><FONT COLOR='#01A982'>Service Prediction</FONT></H1>", unsafe_allow_html=True)

def onClickFunction():
      st.session_state.click = True


############################# test

st.markdown("<h5  color: red;'>Enter The Values</h5>", unsafe_allow_html=True)
col1, col2, col3,col4,col12 = st.columns(5)

with col1:
    OilQual = st.text_input('OilQual',10)
with col2:
    EnginePerf = st.text_input('EnginePerf',10)
with col3:
    NormMileage = st.text_input('NormMileage',10)
with col4:
    TyreWear = st.text_input('TyreWear',10)
with col12:
    HVACwear = st.text_input('HVACwear',10)

    
b_predict = st.button('Predict',on_click=onClickFunction())
OilQual = round(float(OilQual),2)
EnginePerf = round(float(EnginePerf),2)
NormMileage = round(float(NormMileage),2)
TyreWear = round(float(TyreWear),2)
HVACwear = round(float(HVACwear),2)

input_val = pd.DataFrame({
    'OilQual':[OilQual],
    'EnginePerf':[EnginePerf],
    'NormMileage':[NormMileage],
    'TyreWear':[TyreWear],
    'HVACwear':[HVACwear]
})

if "click" not in st.session_state:
    st.session_state.click = False


if st.session_state.click:    ##### Predict Button click session
    X_train = train_data.drop("Service", axis=1)
    y_train = train_data["Service"]

    X_test = test_data.drop("Service", axis=1)
    y_test = test_data["Service"]   
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    rf_best_model = RandomForestClassifier(bootstrap= True,
    ccp_alpha= 0.0,
    class_weight= None,
    criterion= 'gini',
    max_depth= None,
    max_features= 'auto',
    max_leaf_nodes= None,
    max_samples= None,
    min_impurity_decrease= 0.0,
    #min_impurity_split= None,
    min_samples_leaf= 1,
    min_samples_split= 2,
    min_weight_fraction_leaf= 0.0,
    n_estimators= 100,
    n_jobs= None,
    oob_score= False,
    random_state= None,
    verbose= 0,
    warm_start= False)
    
    rf_best_model.fit(X_train, y_train)
    predictions = rf_best_model.predict(X_test)
    rf_accuracy = rf_best_model.score(X_test, y_test)
    rf_report = classification_report(y_test, predictions)
    rf_report = classification_report(y_test, predictions, output_dict=True)
    
    rf_in_pred = rf_best_model.predict(input_val)[0]
    if rf_in_pred == 0:
        rf_in_pred = "No"
    elif rf_in_pred == 1:
        rf_in_pred = "Yes"
    
    
    
    dt_best_model = DecisionTreeClassifier( #bootstrap= True,
    ccp_alpha= 0.0,
    class_weight= None,
    criterion= 'gini',
    max_depth= None,
    max_features= 'auto',
    max_leaf_nodes= None,
    #max_samples= None,
    min_impurity_decrease= 0.0,
    #min_impurity_split= None,
    min_samples_leaf= 1,
    min_samples_split= 2,
    min_weight_fraction_leaf= 0.0,
    #n_estimators= 100,
    #n_jobs= None,
    random_state= None,
    #warm_start= False
    )
    
    dt_best_model.fit(X_train, y_train)
    predictions = dt_best_model.predict(X_test)
    dt_accuracy = dt_best_model.score(X_test, y_test)
    dt_report = classification_report(y_test, predictions)
    dt_report = classification_report(y_test, predictions, output_dict=True)
    dt_in_pred = dt_best_model.predict(input_val)[0]
    
    if dt_in_pred == 0:
        dt_in_pred = "No"
    elif dt_in_pred == 1:
        dt_in_pred = "Yes"
        
        
    lr_best_model = LogisticRegression()
    lr_best_model.fit(X_train, y_train)
    predictions = lr_best_model.predict(X_test)
    lr_accuracy = lr_best_model.score(X_test, y_test)
    lr_report = classification_report(y_test, predictions)
    lr_report = classification_report(y_test, predictions, output_dict=True)
    lr_in_pred = lr_best_model.predict(input_val)[0]
    if lr_in_pred == 0:
        lr_in_pred = "No"
    elif lr_in_pred == 1:
        lr_in_pred = "Yes"



cm1,cm2,cm3 = st.columns(3)
cm1.metric("Random Forest Prediction For Service",rf_in_pred)
cm2.metric("Decision Tree Prediction For Service",dt_in_pred)
cm3.metric("Logistic Refression Prediction For Service",lr_in_pred)



c1,c2 = st.columns(2)
with c1:
    #st.markdown('')
    st.markdown("<h5  color: red;'>Distribution of Data</h5>", unsafe_allow_html=True)
    #with st.expander("Pie Chart Of Investment"):                                         ### Removed expand for pie chart
    values = [OilQual,EnginePerf,NormMileage,TyreWear,HVACwear]
    labels = ['OilQual','EnginePerf','NormMileage','TyreWear','HVACwear']
    colors = ['#F740FF', '#82FFF2','#00C8FF','#FFEB59','#FFBC44']
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        marker=dict(colors=colors),
    ))
    st.plotly_chart(fig)

with c2:
    fig = px.histogram(train_data, x=['OilQual', 'EnginePerf', 'NormMileage', 'TyreWear', 'HVACwear'])

    fig.update_layout(
    title='Train Data Distrubition Plot')
    st.plotly_chart(fig)
    
    

st.markdown("<h2  color: red;'>ML Models Summary</h2>", unsafe_allow_html=True)

rf_rep = pd.DataFrame(rf_report).transpose()
dt_rep = pd.DataFrame(dt_report).transpose()
lr_rep = pd.DataFrame(lr_report).transpose()
s1,s2,s3 = st.columns(3)
with s1:
    st.markdown("<h5  color: red;'>Random Forest Model Summary</h5>", unsafe_allow_html=True)
    st.write(rf_rep)  
with s2:
    st.markdown("<h5  color: red;'>Decision Tree Model Summary</h5>", unsafe_allow_html=True)
    st.write(dt_rep) 
with s3:
    st.markdown("<h5  color: red;'>Logistic Regression Model Summary</h5>", unsafe_allow_html=True)
    st.write(lr_rep)       