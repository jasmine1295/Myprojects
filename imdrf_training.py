
import tqdm
import meddra
#import progressbar
import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
import string
import random
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
#from multi_rake import Rake
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import wordnet
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import URL
from snowflake.sqlalchemy import URL
import difflib

nltk.download('all')
#################################################################################################################################################################################
#################################################################################################################################################################################
###### Rough
def exploding_feedback_data(data,column_name):
  f_file = data#pd.read_excel(data)
  #f_file[['Complaint_Description','IMDRF Annex A','FEEDBACK_CONTENT']]
  #len(f_file['Complaint_Description'].values.tolist())


  def expd(x):
    x = str(x).replace(' ','').split(',')
    x = [item for item in x if item != '']
    return x #"["+str(x)+']'
  f_file[column_name] = f_file[column_name].apply(expd)
  f_file = f_file.explode(column_name)

  def expd(x):
    x = str(x).replace(' ','').split(';')
    x = [item for item in x if item != '']
    return x #"["+str(x)+']'
  f_file[column_name] = f_file[column_name].apply(expd)
  f_file = f_file.explode(column_name)
  return f_file



####### Model retraining
def bulk_retrain(feedback_data,existing_model,new_model_name,code_column_name,feedback_text):
  feedback = pd.read_pickle(existing_model)
  f_file = feedback_data     #pd.read_excel(feedback_file)
  #f_file = imdrf_1.exploding_feedback_data_A(f_file)
  f_file = exploding_feedback_data(f_file,code_column_name)

  ##### Aswin added here below
  def feed(code,text):
    feedback.loc[feedback['Code']==code,'text'] = str(feedback[feedback['Code']==code]['text'].to_list()[0])+' '+str(text)
    return "Data Appended"
  ##### Aswin added here ^up

  p = 0
  for f in f_file[[code_column_name,feedback_text]].values:
    if (str(f[0]).strip() != 'nan') and (str(f[1]).strip() != 'nan'):
      print(p,'= ',f[0])
      try:
        feed(str(f[0]).strip(),f[1])
      except:
        pass
    p+=1

  feedback.to_pickle(new_model_name)


#################################################################################################################################################################################
#################################################################################################################################################################################
def feed(code,text,model):
    feedback = pd.read_pickle(model)
    feedback.loc[feedback['Code']==code,'text'] = str(feedback[feedback['Code']==code]['text'].to_list()[0])+' '+str(text)
    feedback.to_pickle(model)
    return "Data Appended"