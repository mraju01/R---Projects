import pandas as pd
import numpy as np
from keras.utils import pad_sequences
import string
import nltk
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
import re
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding,Flatten,Dense,Conv1D,MaxPooling1D
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator
import plotly.express as px
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from sklearn.preprocessing import OrdinalEncoder