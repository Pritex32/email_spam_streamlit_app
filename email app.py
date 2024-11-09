import  joblib
import numpy as np
import pandas as pd
import streamlit as st


df= pd.read_csv('email.csv')


# duplicates ccheck
df.duplicated().sum()

dfr=df.drop_duplicates(inplace=True) #duplicates removed

df.isnull().sum() # no null values

# coverting the categorical column
from sklearn.preprocessing import LabelEncoder


from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()

x=df['Message']
y=df['Category']

x_v=v.fit_transform(x)

from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_v,y,test_size=0.2,random_state=0)

x_train.shape, x_test.shape,y_train.shape,y_test.shape

#model building 
from sklearn.naive_bayes import MultinomialNB
nb_model=MultinomialNB()

nb_model.fit(x_train,y_train)

nb_model.score(x_train,y_train)

joblib.dump(nb_model,'email_model.joblib')


def email_prediction(msg):
    mat=v.transform([msg])
    return nb_model.predict(mat)[0]


def main():
    st.title('email spam detection'.upper())
    st.info('welcome')

    info=''
    Message=st.text_input('Input your message')
    if st.button('predict'):
        result=nb_model.predict(v.transform([Message]))

        if result == 1:
            info='spam'
        else:
            info='Not a Spam'

            st.success(info)


if __name__ == "__main__":
    main()
