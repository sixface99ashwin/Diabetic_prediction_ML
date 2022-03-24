#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix



df = pd.read_csv('diabetes.csv')

# HEADINGS
# st.title('Diabetes Checkup')
# st.sidebar.header('Patient Data')
# st.subheader('Training Data Stats')
# st.write(df.describe())





for i in df.columns:
    if (i!='Pregnancies' and i!='Outcome'):
        df[i]=df[i].replace(0,round(df[i].mean()))

# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(x.describe())


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# RandomForest MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred=rf.predict(x_test)
user_result = rf.predict(user_data)

#XGboost Classifier Model
model = XGBClassifier() 
model.fit(x_train, y_train) 
user_result1=model.predict(user_data)
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Logistic Regression MOdel
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
user_result2=lr.predict(user_data)


# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'
# VISUALISATIONS
st.title('Visualised Patient Report')

st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)



cm=confusion_matrix(y_test,lr_pred)
cm_rf=confusion_matrix(y_test,rf_pred)
cm_xg=confusion_matrix(y_test,predictions)

# st.write(cm)
# st.write(cm_rf)
# st.write(cm_xg)




# OUTPUT OF LOGISTIC REGRESSION
st.subheader('Your Report By Logistic Regression: ')
output=''
if user_result2[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test,lr.predict(x_test))*100)+'%')

# OUTPUT OF RANDOMFOREST
st.subheader('Your Report By RANDOMFOREST: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

# OUTPUT OF XGBOOST CLASSIFIER
st.subheader('Your Report By XGBOOST CLASSIFIER: ')
output=''
if user_result1[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, model.predict(x_test))*100)+'%')
st.header("Confusion Matrix of Logistic Regression:")
st.write(confusion_matrix(y_test,lr_pred))
st.header("Confusion Matrix of Random Forest:")
st.write(confusion_matrix(y_test,rf_pred))
st.header("Confusion Matrix of XGBOOST CLASSIFIER:")
st.write(confusion_matrix(y_test,predictions))
# st.write("Classification report of Logistic Regression:")
# st.write(classification_report(y_test,lr_pred))
# st.write("Classification report of Random Forest:")
# st.write(classification_report(y_test,rf_pred))
# st.write("Classification report of XGBOOST CLASSIFIER:")
# st.write(classification_report(y_test,predictions))


