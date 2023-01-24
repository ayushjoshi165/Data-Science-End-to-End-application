#We need to install streamlit to run the app to install run the following command

#we will use streamlit to design the User Interface or to host ML model on the internet 

# to run the app after installing required libraries
# 1) open the anaconda power shell command prompt, 
# 2) go to the location where app.py and csv file is located
# 3) run the command -> "streamlit run app.py"
# 4) The app will start in the browser
# 5) I will attach the screenshot of terminal in the ppt as well. 

#pip install streamlit(to install streamlit)

#now we need to install or import pandas library that lets you easily use data structures
# and data analysis tools for the Python programming language

#pip install pandas

#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

#mathplot to draw the graphical presentation
import matplotlib.pyplot as plt
#import plotly.figure_factory as ff

#SciKit Learn includes everything from dataset manipulation to processing metrics. 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns



df = pd.read_csv(r"D:\Desktop\diabetes.csv")

# to create various elements like heading , sidebar and title we use streamlit function like below
st.title('Diabetes Predication Report')
st.sidebar.header('Patient Data')
st.subheader('Training Data ')
st.write(df.describe())


# x data shuold not contains the output column so we have removed it from the data 
x = df.drop(['Outcome'], axis = 1)

# y contains the output 
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# FUNCTION
def user_report():

  #here we have used the slider and passed minimum maximum and starting point of pregnancies slider 
  pregnancies = st.sidebar.slider('Select Pregnancies', 0,17, 3 )

  # #here we have used the slider and passed minimum maximum and starting point of glucose slider  
  glucose = st.sidebar.slider('Select Glucose', 0,200, 120 )

   #here we have used the slider and passed minimum maximum and starting point of bp slider 
  bp = st.sidebar.slider('Select Blood Pressure', 0,122, 70 )

   #here we have used the slider and passed minimum maximum and starting point of skinthickness slider 
  skinthickness = st.sidebar.slider('Select Skin Thickness', 0,100, 20 )

   #here we have used the slider and passed minimum maximum and starting point of insulin slider 
  insulin = st.sidebar.slider('Select Insulin', 0,846, 79 )

   #here we have used the slider and passed minimum maximum and starting point of bmi slider 
  bmi = st.sidebar.slider('Select BMI', 0,67, 20 )

   #here we have used the slider and passed minimum maximum and starting point of dpf slider 
  dpf = st.sidebar.slider('Select Diabetes Pedigree Function', 0.0,2.4, 0.47 )

   #here we have used the slider and passed minimum maximum and starting point of age slider 
  age = st.sidebar.slider('Select Age', 21,88, 33 )

  
#created dictionary to store all of my values
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

  #repost data is converted into python dataframe 
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA is stored in the user_data variable
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# Created a variable that is a random forest classifier
rf  = RandomForestClassifier()


# Now we show the accuracy of our model
# st.subheader('Accuracy: ')
# st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

# Now we fit our data to the model
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



## VISUALISATIONS
st.title('Visualised Patient Report')



# COLOR FUNCTION 

# here we are representing the position as Red if the patient is diabitic and blue if patient is not diabitic
if user_result[0]==0:
 color = 'blue'
else:
 color = 'red'


# Age vs Pregnancies graph against the patient vs others 
st.header('Pregnancy count Graph (Others vs Patient)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose graph against the patient vs others 
st.header('Glucose Value Graph (Others vs Patient)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp graph against the patient vs others 
st.header('Blood Pressure Value Graph (Others vs Patient)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St graph against the patient vs others 
st.header('Skin Thickness Value Graph (Others vs Patient)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin graph against the patient vs others 
st.header('Insulin Value Graph (Others vs Patient)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI graph against the patient vs others 
st.header('BMI Value Graph (Others vs Patient)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf graph against the patient vs others 
st.header('DPF Value Graph (Others vs Patient)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT
st.subheader('Patient Report: ')
output=''
if user_result[0]==0:
 output = 'Patient is not Diabetic'
else:
 output = 'Patient is Diabetic'
st.title(output)

# here we have calculated the accuracy score for the model 
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')
