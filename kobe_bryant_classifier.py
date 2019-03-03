import numpy as np
import scipy as sp
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold   #from sklearn.model_selection import KFold




# best_n: 100
# best_m: 10
#
# Mean Absolute Train Error: 0.3 degrees.
# Train Accuracy: 70.18 %.
# Mean Absolute Test Error: 0.33 degrees.
# Test Accuracy: 67.08 %.




filename= "data.csv"
# raw = pd.read_csv(filename)

shot_zone_area_dict = {'Center(C)': 1, 'Left Side(L)': 4, 'Right Side(R)': 4, 'Left Side Center(LC)': 7, 'Right Side Center(RC)': 7, 'Back Court(BC)': 10}

def process_data():
    raw = pd.read_csv(filename)
    # drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
             # 'matchup', 'game_event_id', 'game_id', 'game_date']
    # drops = ['team_id', 'team_name', 'matchup', 'game_id', 'game_date']
    drops = ['team_id', 'team_name', 'game_event_id', 'game_id', 'game_date', 'shot_id']
    for drop in drops:
        raw = raw.drop(drop, 1)

    # categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'season']
    # categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'season', 'shot_id', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
    #          'matchup', 'game_event_id', 'game_date']
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', 'opponent', 'season']
    for var in categorical_vars:
        raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
        raw = raw.drop(var, 1)

    for i, row in enumerate(raw.itertuples(), 1):
        if "@" in row.matchup:
            raw.set_value(row.Index, 'matchup', 0)
        else:
            raw.set_value(row.Index, 'matchup', 1)

    df = raw[pd.notnull(raw['shot_made_flag'])]
    indexOfNull = raw[raw['shot_made_flag'].isnull()].index.tolist()

    return raw, df, indexOfNull


def split_data(raw, df, indexOfNull):
    per_test =  round((15 * len(df))/100)
    n =[0 for i in range(per_test)]

    m = len(indexOfNull)-1
    for z in range(per_test):
        c = indexOfNull[m]+1
        x = raw.iloc[c]
        #print(pd.notnull(x['shot_made_flag']))
        flg = 0
        if(pd.isnull(x['shot_made_flag'])):
            #print(c)
            flg =1
            while flg==1:
                c = c+1
                x = raw.iloc[c]
                if pd.notnull(x['shot_made_flag']):
                    flg =0
        n[z] = c
        m=m-1;



    test_comp = raw.iloc[n]
    test = test_comp.drop('shot_made_flag', 1)
    test_y = test_comp['shot_made_flag']
    df = raw.drop(raw.index[n])
    df= df[pd.notnull(df['shot_made_flag'])]
    # separate df into explanatory and response variables
    train = df.drop('shot_made_flag', 1)
    train_y = df['shot_made_flag']

    return train, train_y, test, test_y


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def find_RandomForest_parameters(train, plot=False):
    # find the best n_estimators for RandomForestClassifier
    print('Finding best n_estimators for RandomForestClassifier...')
    min_score = 100000
    best_n = 0
    scores_n = []
    range_n = np.logspace(0,2,num=3).astype(int)

    for n in range_n:
        print("the number of trees : {0}".format(n))
        t1 = time.time()

        rfc_score = 0.
        rfc = RandomForestClassifier(n_estimators=n)
        for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
            rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
            #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
            pred = rfc.predict(train.iloc[test_k])
            rfc_score += logloss(train_y.iloc[test_k], pred) / 10
        scores_n.append(rfc_score)
        if rfc_score < min_score:
            min_score = rfc_score
            best_n = n

        t2 = time.time()
        print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2-t1))
    print(best_n, min_score)

    # find best max_depth for RandomForestClassifier
    print('Finding best max_depth for RandomForestClassifier...')
    min_score = 100000
    best_m = 0
    scores_m = []
    range_m = np.logspace(0,2,num=3).astype(int)
    for m in range_m:
        print("the max depth : {0}".format(m))
        t1 = time.time()

        rfc_score = 0.
        rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
        for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
            rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
            #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
            pred = rfc.predict(train.iloc[test_k])
            rfc_score += logloss(train_y.iloc[test_k], pred) / 10
        scores_m.append(rfc_score)
        if rfc_score < min_score:
            min_score = rfc_score
            best_m = m

        t2 = time.time()
        print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2-t1))
    print(best_m, min_score)

    if (plot):
        plot_RandomForest_parameters(range_n, scores_n, range_m, scores_m)

    return best_n, best_m


def plot_RandomForest_parameters(range_n, scores_n, range_m, scores_m):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(range_n, scores_n)
    plt.ylabel('score')
    plt.xlabel('number of trees')

    plt.subplot(122)
    plt.plot(range_m, scores_m)
    plt.ylabel('score')
    plt.xlabel('max depth')

    plt.show()


def run_RandomForest(train, train_y, test, best_n, best_m):
    model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
    model.fit(train, train_y)
    # pred_prob = model.predict_proba(train)
    # pred_tes_probt = model.predict_proba(test)


    pred_train = model.predict(train)
    # pred = model.predict(submission)
    pred_test = model.predict(test)

    return pred_train, pred_test


def predictions_result(pred_train, train_y, pred_test, test_y):
    # Calculate the absolute errors
    errors = abs(pred_train - train_y)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Train Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Train Accuracy:', round(accuracy, 2), '%.')


    # Calculate the absolute errors
    errors1 = abs(pred_test - test_y)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Test Error:', round(np.mean(errors1), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors1)
    # Calculate and display accuracy
    accuracy1 = 100 - np.mean(mape)
    print('Test Accuracy:', round(accuracy1, 2), '%.')


'''
MAIN
'''
if __name__ == '__main__':
    raw, df, indexOfNull = process_data()
    train, train_y, test, test_y = split_data(raw, df, indexOfNull)
    best_n, best_m = find_RandomForest_parameters(train)
    pred_train, pred_test = run_RandomForest(train, train_y, test, best_n, best_m)
    predictions_result(pred_train, train_y, pred_test, test_y)
