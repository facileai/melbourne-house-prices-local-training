from explainerdashboard import RegressionExplainer, ExplainerDashboard
import pandas as pd
import pickle
from fastai.basics import *
import json
from sklearn.model_selection import StratifiedShuffleSplit

def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))

def load_dataset(data_set_path,n_samples=1000):

    print('Loading data from {}'.format(data_set_path))
    df = pd.read_csv(data_set_path, low_memory=False)
    df['splitter'] = df['Suburb']+''+df['Postcode'].astype(str)
    splitter = 'splitter'

    #identify splitters that appear once and set them aside
    df_group_splitter = df.groupby([splitter]).agg({'Postcode':['count']})
    df_group_splitter.columns = ['count_splitter']
    df_group_splitter.reset_index( inplace = True)
    df = join_df(df,df_group_splitter,splitter)

    cond = df['count_splitter'] > 1
    len_uniq_splitter = (~cond).sum()

    if n_samples > len_uniq_splitter:
        n_samples = n_samples - len_uniq_splitter

    
    df.drop(['count_splitter'],axis=1, inplace=True)

    cols = list(df.columns.values)
    cols.remove(splitter)
    
    X = df.loc[cond,cols]
    y = df.loc[cond, splitter]

    splits = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=42)
   
    for _, test_index in splits.split(X, y):
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
    

    X_test = pd.concat([X_test,df.loc[~cond,cols]],ignore_index=True)

    return X_test

def load_TabPandas(fname):
    "Load in a `TabularPandas` object from `fname`"
    distrib_barrier()
    res = pickle.load(open(fname, 'rb'))
    return res


if __name__ == '__main__':
   
    #Explainer dashboard

    valid_df = load_dataset('./data/mel-valid.csv',n_samples=350)

    with open('artifacts/features.txt') as json_file:
            features = json.load(json_file)
    
    cont_nn = features['cont']
    cat_nn = features['cat']
            
    # feature_descriptions = {
    #     "Elapsed": "Time Elapsed",
    #     "fips": "fips code to uniquely identify regions within a county",
    #     "Year": "Year",
    # }

    with open('./artifacts/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('./artifacts/data-proc.pkl', 'rb') as preproc_file:
        preproc = pickle.load(preproc_file)

    val_nn = preproc.train.new(valid_df)
    val_nn.process()
    
    X_valid,Y_valid = val_nn.train.xs,val_nn.train.y

    print(X_valid.shape,Y_valid.shape)

    explainer = RegressionExplainer(model, X_valid, Y_valid, units = "$")

    ExplainerDashboard(explainer,shap_interaction=False, no_permutations=True, check_additivity=False,feature_perturbation='interventional').run(port=8082)