import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

from wordcloud import WordCloud, ImageColorGenerator

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from gensim.utils import simple_preprocess

STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def save_object(obj, path):
    pickle.dump(obj, open(path, 'wb'))
    
def load_object(path):
    return pickle.load(open(path, 'rb'))

def nlp_preprocess(text, use_stemmer = False):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            if not use_stemmer:
                result.append(lemmatizer.lemmatize(token, pos='v'))
            else:
                result.append(stemmer.stem(lemmatizer.lemmatize(token, pos='v')))
    return " ".join(result)

def generate_wordcloud(text):
    text_to_generate = text
    if type(text) == list:
        text_to_generate = " ".join([" ".join(i) for i in text]) if type(text[0]) == list else " ".join(text)
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = 'white', collocations = False).generate(text_to_generate)
    plt.figure(figsize = (13.8, 6))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()
    
    

#############################################   DATAFRAME PREPROCESSING    #############################################
yes_no_map_cap = {"Yes": 1, "No": 0, "nan": np.nan}
yes_no_map_lower = {"yes": 1, "no": 0, "nan": np.nan}
begscl_map = {'I always received my schooling in English': 1, 'age 2-4': 2, 'age 5-7': 3, 
              'age 8-10': 4, 'age 11-13': 5, 'age 14-17': 6, 'age 18-21': 7,'after age 21': 8, "nan": np.nan}
parents_map = {'No': 1, 'Yes, one parent': 2, "Yes, both parents": 3, 'nan': np.nan}
resid_map = {"On-campus": 1, "Off-campus, but less than one hour away": 2, "Off-campus, and more than one hour away": 3, "nan": np.nan}
courses_map = {'0': 1, '1': 2, '2': 3, '3+':4, "4": np.nan, 'nan': np.nan}
courses_imp_map = {"Most important": 4, "Second-most important": 3, "Third-most important": 2, "Fourth-most important": 1, "nan": np.nan}
courses_int_map = {"Most interesting": 4, "Second-most interesting": 3, "Third-most interesting": 2, "Fourth-most interesting": 1, "nan": np.nan}
aca_map = {'Every week': 6, 'Never': 1, 'Once a month': 4, 'Once a quarter':2, 'Twice a month':3, 'Twice a quarter':5 , "nan": np.nan}
sp_map = { 'Neither agree nor disagree':3, 'Strongly agree':5, 'Strongly disagree':1, '4': 4, '2':2, "nan": np.nan}
canv_map = {'Important':4, 'Somewhat unimportant':3, 'Unimportant':2, 'Very important':5, 'Very unimportant': 1, "nan": np.nan}
studyplan_chg_map = {'I never had a study plan': 1, 'No, I stuck to my plan': 0, 
                     'Yes, I changed my study plan a bit':2, 'Yes, I changed my study plan a lot':3,'nan': np.nan}
sex_map = {'Female': 0, 'Male': 1, 'nan': np.nan}
perfown_map = {'About the same':1, 'Better':2, "I don't know":4, 'Worse':3, 'nan':np.nan}
perforce_map = {'I am about the same as other students':2,
 'I am less smart than other students':3, 'I am smarter than other students':1,
 "Others won't have a way of judging whether I am smart in my class":4, 'nan':np.nan}
grade_scale_map = {'A+': 13, 'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8, 'C+': 7, 'C': 6, 
                   'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'F': 1, 'NA': np.nan, "nan": np.nan}

act_cluster_label_map = {
    'oact': {0: 'General', 1: 'Work', 2: 'Another Course', 3: 'Personal'},
    'cact': {}
}

COL_MAP_LIST = [yes_no_map_cap, yes_no_map_lower,begscl_map, parents_map, resid_map, courses_map, courses_imp_map, 
           courses_int_map, aca_map, sp_map, canv_map, studyplan_chg_map, sex_map, perfown_map, perforce_map, grade_scale_map]

grab_vars = lambda key: [j for v in VAR_MAP[key].values() for j in list(v)]

def separate_quant_fr(df, var_map):
    new_var_map = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for key in var_map:
        data_types = df[grab_vars(key)].dtypes
        for construct in VAR_MAP[key]:
            for var in VAR_MAP[key][construct]:
                if data_types[var] == np.dtype('float64'):
                    new_var_map[key]["Quant"][construct].add(var)
                else:
                    new_var_map[key]["FR"][construct].add(var)
    original_all_var =  [var for k in VAR_MAP for var_list in VAR_MAP[k].values() for var in var_list]
    reformated_all_var = [var for k in new_var_map for t in ["Quant", "FR"] for var_list in new_var_map[k][t].values() for var in var_list ]
    assert(set(original_all_var) == set(reformated_all_var))
    return new_var_map

def convert_fr_col(df, num_uniq = 16, verbose = True):
    
    def map_column(df, v):
        mapped, col_type = False, "str"
        try:
            uniq_val, col_type = np.unique(df[v].astype('float64')).tolist(), "float"
        except:
            uniq_val = np.unique(df[v].astype('str')).tolist()
            
        for col_map in COL_MAP_LIST:
            if len(set(uniq_val).intersection(set(col_map.keys()))) == len(uniq_val):
                df[v] = df[v].map(col_map)
                mapped = True
                break
                
        if not mapped and verbose:
            print("Cannot find map for variable {}(type = {}), uniq vals are {}".format(v, col_type, uniq_val))
        return df
        
    for v in ALL_NON_QUANT_VAR:
        if df[ALL_NON_QUANT_VAR].nunique()[v] <= num_uniq:
            df = map_column(df, v)
        elif verbose:
            print("Var {} has more than {} unique vals".format(v, num_uniq))
    return df

def get_fr_cluster_label(df):
    
    def label_activity(df, cluster_pkl, act_type):
        vectorizer, classifier = cluster_pkl[act_type]
        col_names = [c for c in df.columns if "oact" in c and "compx" not in c] if act_type == "oact" else \
                    ["dcact{}".format(i) for i in range(1, 31)] if act_type == "cact" else None
        col_names = list(set(col_names).intersection(ALL_NON_QUANT_VAR))
        for col in col_names:
            act_data = [nlp_preprocess(act, use_stemmer=True) for act in df[col].fillna('').tolist()]
            df[col] = classifier.predict(vectorizer.transform(act_data)).tolist()
        return df
    
    cluster_pkl = load_object("../data/act_list_cluster.pkl")
    for act_type in ["oact", 'cact']:
        df = label_activity(df, cluster_pkl, act_type)
    return df

def encode_target_vars(df):
    target_vars = ["gr_revq1", "gr_revq2", "gr_revq3", "gr_revq4", "gr_revq5", "gr_exam1", "gr_exam2", "gr_exam3"]
    all_target_vars = []
    for var in target_vars:
        quantile_encodings = []
        df[var] = df[var].fillna(0)
        
        bot = np.quantile(df[var], .25)
        med = np.quantile(df[var], .50)
        top = np.quantile(df[var], .75)
        
        for val in df[var]:
            if val <= bot:
                quantile_encodings.append(1)
            elif val > bot and val <= med:
                quantile_encodings.append(2)
            elif val > med and val <= top:
                quantile_encodings.append(3)
            else:
                quantile_encodings.append(4)
                
        df[var + "_quantile"] = quantile_encodings
        all_target_vars.append(var)
        all_target_vars.append(var + "_quantile")
    return df

def preprocess_df(df, verbose = False):
    proc_df = df[df["pre_studyinterest"] == "Yes"].reset_index(drop = True).copy()
    proc_df = encode_target_vars(proc_df)
    proc_df = convert_fr_col(proc_df, verbose = verbose)
    proc_df = get_fr_cluster_label(proc_df)
    return proc_df

df = pd.read_csv("../data/class_data.csv")
VAR_MAP = load_object("../data/var_map.pkl")
NEW_VAR_MAP = separate_quant_fr(df, VAR_MAP)

ALL_VAR = [var for k in VAR_MAP for var_list in VAR_MAP[k].values() for var in var_list]
ALL_QUANT_VAR = [var for k in NEW_VAR_MAP for var_list in NEW_VAR_MAP[k]["Quant"].values() for var in var_list]
ALL_NON_QUANT_VAR = [var for k in NEW_VAR_MAP for var_list in NEW_VAR_MAP[k]["FR"].values() for var in var_list]