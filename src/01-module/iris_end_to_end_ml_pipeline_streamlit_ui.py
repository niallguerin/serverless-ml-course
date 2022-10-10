#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification with Scikit-Learn
# 
# ![Iris](https://github.com/featurestoreorg/serverless-ml-course/raw/main/src/01-module/assets/iris.png)
# 
# 
# In this notebook we will, 
# 
# 1. Load the Iris Flower dataset into Pandas from a CSV file
# 2. Split training data into train and test sets (one train/test set each for both the features and labels)
# 3. Train a KNN Model using SkLearn
# 4. Evaluate model performance on the test set
# 5. Visually query the model "predictive analytics"

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import streamlit as st

# We are downloading the 'raw' iris data. We explicitly do not want transformed data, reading for training. 
# 
# So, let's download the iris dataset, and preview some rows. 
# 
# Note, that it is 'tabular data'. There are 5 columns: 4 of them are "features", and the "variety" column is the **target** (what we are trying to predict using the 4 feature values in the target's row).

# In[2]:


iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
# iris_df.sample(10)


# We can see that our 3 different classes of iris flowers have different *petal_lengths* 
# (although there are some overlapping regions between Versicolor and the two other varieties (Setoas, Virginica))

# In[3]:


sns.set(style='white', color_codes=True)

sns.boxplot(x='variety', y='sepal_length', data=iris_df)


# In[4]:


sns.set(style='white', color_codes=True)

sns.boxplot(x='variety', y='sepal_width', data=iris_df)


# In[5]:


sns.set(style='white', color_codes=True)

sns.boxplot(x='variety', y='petal_length', data=iris_df)


# In[6]:


sns.set(style='white', color_codes=True)

sns.boxplot(x='variety', y='petal_width', data=iris_df)


# We need to split our DataFrame into two Dataframes. 
# 
# * The **features** DataFrame will contain the inputs for training/inference. 
# * The **labels** DataFrame will contain the target we are trying to predict.
# 
# Note, that the ordering of the rows is preserved between the features and labels. For example, 'row 40' in the **features** DataFrame contains the correct features for 'row 40' in the **labels** DataFrame. That is, the row index acts like a common "join key" between the two DataFrames.

# Split the DataFrame into 2: one DataFrame containing the *features* and one containing the *labels*.

# In[7]:


features = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
labels = iris_df[["variety"]]
# features


# In[8]:


# labels


# We can split our features and labels into a **train_set** and a **test_set**. You split your data into a train_set and a test_set, because you want to train your model on only the train_set, and then evaluate its performance on data that was not seen during training, the test_set. This technique helps evaluate the ability of your model to accurately predict on data it has not seen before.
# 
# This looks as follows:
# 
# * **X_** is a vector of features, so **X_train** is a vector of features from the **train_set**. 
# * **y_** is a scale of labels, so **y_train** is a scalar of labels from the **train_set**. 
# 
# Note: a vector is an array of values and a scalar is a single value.
# 
# Note: that mathematical convention is that a vector is denoted by an uppercase letter (hence "X") and a scalar is denoted by a lowercase letter (hence "y").
# 
# **X_test** is the features and **y_test** is the labels from our holdout **test_set**. The **test_set** is used to evaluate model performance after the model has been trained.
# 

# In[9]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2)
# y_train


# We can see that our original lables (**y_train** and **y_test**) are categorical variables. 
# 
# We could transform the label from a categorical variable (a string) into a numerical variable (an int). Many machine learning training algorithms only take numerical values as inputs for training (and inference).
# However, our ML algorithm, KNeighborsClassifier, works with categorical variables as labels.
# 
# A useful exercise here is to use Scikit-Learn's LabelEncoder to transform the labels to a numerical representation.

# Now, we can fit a model to our features and labels from our training set (**X_train** and **y_train**). Fitting a model to a dataset is more commonly called "training a model".

# In[10]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train.values.ravel())


# Now, we have trained our model. We can evaluate our model on the **test_set** to estimate its performance.
# 
# Scikit-Learn's KNeighborsClassifier take a DataFrame as input and returns a list of predictions.
# 
# Notice that for each input feature vector (containing our 4 features: sepal_length, sepal_width, petal_length, petal_width), the model returns a prediction of the type of flower.

# In[11]:


y_pred = model.predict(X_test)
# y_pred


# We can report on how accurate these predictions (**y_pred**) are compared to the labels (the actual results - **y_test**).

# In[12]:


from sklearn.metrics import classification_report

metrics = classification_report(y_test, y_pred, output_dict=True)
# print(metrics)


# In[13]:


from sklearn.metrics import confusion_matrix

results = confusion_matrix(y_test, y_pred)
# print(results)


# Notice in the confusion matrix results that we have 1 or 2 incorrect predictions.
# We have only 30 flowers in our test set - **y_test**.
# Our model predicted 1 or 2 flowers were of type "Virginica", but the flowers were, in fact, "Versicolor".

# In[14]:


from matplotlib import pyplot

df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                     ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

sns.heatmap(df_cm, annot=True)


# In[15]:


# get_ipython().system('pip install gradio --quiet')
# get_ipython().system('pip install typing-extensions==4.3.0')


# In[16]:


import gradio as gr
import numpy as np
from PIL import Image
import requests

def iris(sepal_length, sepal_width, petal_length, petal_width):
    input_list = []
    input_list.append(sepal_length)
    input_list.append(sepal_width)
    input_list.append(petal_length)
    input_list.append(petal_width)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     flower_url = "https://repo.hops.works/master/hopsworks-tutorials/data/" + res[0] + ".png"
    flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    img = Image.open(requests.get(flower_url, stream=True).raw)            
    return img
        
# demo = gr.Interface(
#     fn=iris,
#     title="Iris Flower Predictive Analytics",
#     description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
#     allow_flagging="never",
#     inputs=[
#         gr.inputs.Number(default=1.0, label="sepal length (cm)"),
#         gr.inputs.Number(default=1.0, label="sepal width (cm)"),
#         gr.inputs.Number(default=1.0, label="petal length (cm)"),
#         gr.inputs.Number(default=1.0, label="petal width (cm)"),
#         ],
#     outputs=gr.Image(type="pil"))
#
# demo.launch(share=True)

# streamlit ui experiment for fun - reuse Serverless ML code from function and just follow streamlit docs
# one of the optional tasks for week1 homework exercise for this project. Add a streamlit ui versus a gradio ui shown in default notebook
# streamlit according to their own docs is not workable inside jupyter notebook so ran this as a .python file per section.io tutorial and it runs fine
# modified the url path to point to my github forked repo from featurestore.org main serverless-ml course for the original python notebook
# commented out dataframe print options so it just renders streamlit ui only and not the dataframe sections
st.title("Iris Flower Predictive Analytics - Streamlit UI")
st.subheader("Experiment with sepal/petal lengths/widths to predict which flower it is.")
sepal_length = st.number_input("Enter sepal length")
sepal_width = st.number_input("Enter sepal width")
petal_length = st.number_input("Enter petal length")
petal_width = st.number_input("Enter petal width")

input_list = []
input_list.append(sepal_length)
input_list.append(sepal_width)
input_list.append(petal_length)
input_list.append(petal_width)
res = model.predict(np.asarray(input_list).reshape(1, -1))
# refactored to use my github repo copy and not the class repo for featurestore
flower_url = "https://raw.githubusercontent.com/niallguerin/serverless-ml-course/main/src/01-module/assets/" + res[
    0] + ".png"
img = Image.open(requests.get(flower_url, stream=True).raw)
st.image(img, caption="The predicted flower is:")

# Web References
https://www.section.io/engineering-education/streamlit-ui-tutorial/
https://docs.streamlit.io/library/api-reference/media/st.image
