import numpy as np
import scipy as sp
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold   #from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



# RANDOM FOREST
# best_n: 100
# best_m: 10
#
# Mean Absolute Train Error: 0.3 degrees.
# Train Accuracy: 70.18 %.
# Mean Absolute Test Error: 0.33 degrees.
# Test Accuracy: 67.08 %.

# LOGISTIC REGRESSION
# solver = liblinear
# Mean Absolute Train Error: 0.32 degrees.
# Train Accuracy: 68.32 %.
# Mean Absolute Test Error: 0.32 degrees.
# Test Accuracy: 67.81 %.
# solver = lbfgs
# Mean Absolute Train Error: 0.32 degrees.
# Train Accuracy: 67.83 %.
# Mean Absolute Test Error: 0.33 degrees.
# Test Accuracy: 66.67 %.

# NAIVE BAYES
# Mean Absolute Train Error: 0.38 degrees.
# Train Accuracy: 62.46 %.
# Mean Absolute Test Error: 0.37 degrees.
# Test Accuracy: 63.06 %.


filename= "data.csv"
# raw = pd.read_csv(filename)

shot_zone_area_dict = {'Center(C)': 1, 'Left Side(L)': 4, 'Right Side(R)': 4, 'Left Side Center(LC)': 7, 'Right Side Center(RC)': 7, 'Back Court(BC)': 10}

def process_data():
    raw = pd.read_csv(filename)

    for i, row in enumerate(raw.itertuples(), 1):
        if "@" in row.matchup:
            raw.set_value(row.Index, 'matchup', 0)
        else:
            raw.set_value(row.Index, 'matchup', 1)

        raw.set_value(row.Index, 'seconds_remaining', row.seconds_remaining + row.minutes_remaining * 60)

    # raw = raw.drop(columns='minutes_remaining')
    unsorted_raw = raw
    raw = raw.sort_values(['season', 'shot_id'], ascending=[True, True])



    # drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
             # 'matchup', 'game_event_id', 'game_id', 'game_date']
    # drops = ['team_id', 'team_name', 'matchup', 'game_id', 'game_date']
    drops = ['team_id', 'team_name', 'game_event_id', 'game_id', 'game_date', 'shot_id', 'minutes_remaining', 'lat', 'lon']
    for drop in drops:
        raw = raw.drop(drop, 1)

    # categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'season']
    # categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'season', 'shot_id', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
    #          'matchup', 'game_event_id', 'game_date']
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', 'opponent']#, 'season'] # remove "season" when split the data
    for var in categorical_vars:
        raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
        raw = raw.drop(var, 1)

    # df = raw[pd.notnull(raw['shot_made_flag'])]
    # indexOfNull = raw[raw['shot_made_flag'].isnull()].index.tolist()

    # raw.to_csv("data_processed.csv", sep=',')

    return raw, unsorted_raw


def split_for_experiments(raw):
    df = raw[pd.notnull(raw['shot_made_flag'])]
    seasons_dict = {k: v for k, v in df.groupby('season')}

    seasons_loc_dict = {}
    for k, v in seasons_dict.items():
        temp_dict = {(k, k2): v2 for k2, v2 in v.groupby('matchup')}
        seasons_loc_dict.update(temp_dict)

    seasons_mode_dict = {}
    for k, v in seasons_dict.items():
        temp_dict = {(k, k2): v2 for k2, v2 in v.groupby('playoffs')}
        seasons_mode_dict.update(temp_dict)

    return seasons_dict, seasons_loc_dict, seasons_mode_dict


def split_in_blocks(seasons_dict, type):
    blocks_dict = {}

    if type == 'season':
        for k, v in seasons_dict.items():
            block_dim = len(v.index) // 10
            blocks = []
            for i in range(10):
                start = i * block_dim
                end = (i + 1) * block_dim

                if i != 9:
                    block = v.iloc[start:end]
                else:
                    block = v.iloc[start:]

                blocks.append(block)
            blocks_dict[k] = blocks
    elif type == 'location':
        home = []
        away = []
        for k, v in seasons_dict.items():
            block_dim = len(v.index) // 5
            blocks = []
            for i in range(5):
                start = i * block_dim
                end = (i + 1) * block_dim

                if i != 4:
                    block = v.iloc[start:end]
                else:
                    block = v.iloc[start:]

                blocks.append(block)

            if k[1] == 1:
                home += blocks
            else:
                away += blocks

        blocks_dict['home'] = home
        blocks_dict['away'] = away
    elif type == 'mode':
        season = []
        playoff = []
        for k, v in seasons_dict.items():
            if k[1] == 1:
                n = 2
            else:
                n = 8

            block_dim = len(v.index) // n
            blocks = []
            for i in range(n):
                start = i * block_dim
                end = (i + 1) * block_dim

                if i != n-1:
                    block = v.iloc[start:end]
                else:
                    block = v.iloc[start:]

                blocks.append(block)

            if k[1] == 1:
                playoff += blocks
            else:
                season += blocks

        blocks_dict['playoff'] = playoff
        blocks_dict['season'] = season

    return blocks_dict


def get_all_testing_points(raw):
    df = raw[pd.notnull(raw['shot_made_flag'])]
    indexOfNull = raw[raw['shot_made_flag'].isnull()].index.tolist()

    per_test =  round((15 * len(df))/100)
    n =[0 for i in range(per_test)]

    m = len(indexOfNull)-1
    for z in range(per_test):
        c = indexOfNull[m]+1

        if c not in raw.index:      # check the end of the chunk
            break

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

    # print (n[0], n[3854])

    return n

def split_data(block, all_points, type='default'):

    if type == 'default':
        block = pd.concat([block, pd.get_dummies(block['season'], prefix='season')], 1)
        block = block.drop('season', 1)
    else:
        block = block.drop('season', 1)
        if type == 'location':
            block = block.drop('matchup', 1)
        elif type == 'mode':
            block = block.drop('playoffs', 1)

    indices = block.index.tolist()
    test_points = np.intersect1d(indices, all_points)
    # print(indices)
    # print(all_points)

    # n =[i for i in test_points]
    #
    # print (n)

    test_comp = block.loc[test_points]
    test = test_comp.drop('shot_made_flag', 1)
    test_y = test_comp['shot_made_flag']
    df = block.drop(test_points)
    df= df[pd.notnull(df['shot_made_flag'])]
    train = df.drop('shot_made_flag', 1)
    train_y = df['shot_made_flag']

    return train, train_y, test, test_y

def split_by_field(df):
    train = pd.DataFrame()
    train_y = pd.DataFrame()

    if not df.empty:
        train = df.drop('shot_made_flag', 1)
        train_y = df['shot_made_flag']

    return train, train_y


# def split_data(raw):
#     df = raw[pd.notnull(raw['shot_made_flag'])]
#     indexOfNull = raw[raw['shot_made_flag'].isnull()].index.tolist()
#
#     per_test =  round((15 * len(df))/100)
#     n =[0 for i in range(per_test)]
#
#     m = len(indexOfNull)-1
#     for z in range(per_test):
#         c = indexOfNull[m]+1
#
#         if c not in raw.index:      # check the end of the chunk
#             break
#
#         x = raw.iloc[c]
#         #print(pd.notnull(x['shot_made_flag']))
#         flg = 0
#         if(pd.isnull(x['shot_made_flag'])):
#             #print(c)
#             flg =1
#             while flg==1:
#                 c = c+1
#                 x = raw.iloc[c]
#                 if pd.notnull(x['shot_made_flag']):
#                     flg =0
#         n[z] = c
#         m=m-1;
#
#
#
#     test_comp = raw.iloc[n]
#     test = test_comp.drop('shot_made_flag', 1)
#     test_y = test_comp['shot_made_flag']
#     df = raw.drop(raw.index[n])
#     df= df[pd.notnull(df['shot_made_flag'])]
#     # separate df into explanatory and response variables
#     train = df.drop('shot_made_flag', 1)
#     train_y = df['shot_made_flag']
#
#     return train, train_y, test, test_y





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


def run_LogisticRegression(train, train_y, test):
    logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
    logreg.fit(train, train_y)

    pred_train = logreg.predict(train)
    pred_test = logreg.predict(test)

    return pred_train, pred_test

def run_NaiveBayes(train, train_y, test):
    gnb = GaussianNB()
    gnb.fit(train, train_y)

    pred_train = gnb.predict(train)
    pred_test = gnb.predict(test)

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
'''
if __name__ == '__main__':
    raw, unsorted_raw = process_data()
    seasons_dict, seasons_loc_dict, seasons_mode_dict = split_for_experiments(raw)
    seasons_blocks_dict = split_in_blocks(seasons_dict, 'season')
    seasons_loc_blocks_dict = split_in_blocks(seasons_loc_dict, 'location')
    seasons_mode_blocks_dict = split_in_blocks(seasons_mode_dict, 'mode')
    # print(seasons_blocks_dict)


    all_points = get_all_testing_points(unsorted_raw)
    # print(all_points)






    ## KOMAL: - you have to do something like this in order to get the sets for a block
    ##        - this is just for the first block from the first season
    # for seasons

    temp_block = pd.DataFrame()

    for k,v in seasons_blocks_dict.items():
        copy = v.copy()
        # print (copy)
        # print(copy[0].index[0], copy[9].index[40])
        for i in range(len(copy)):
            print("BLOCK", i)
            # print("Original block length:", len(copy[i].index))

            temp_train, temp_train_y = split_by_field(temp_block)

            train, train_y, test, test_y = split_data(copy[i], all_points, 'season')
            # print("After split length:", len(train.index), len(test.index))
            # print(train, train_y, test, test_y)

            train = pd.concat([temp_train, train])
            train_y = pd.concat([temp_train_y, train_y])
            # print("After concat length:", len(train.index), len(test.index))
            # print(train, train_y, test, test_y)
            print(train, test)

            for j in test.index:
                copy[i].xs(i)['shot_made_flag'] = #the result after prediction

            copy[i] = copy[i].drop('season', 1)
            temp_block = pd.concat([temp_block, copy[i]])
        # break

    # for home&away
    # for k,v in seasons_loc_blocks_dict.items():
    #     train, train_y, test, test_y = split_data(v[0], all_points, 'location')
    #     print(train, train_y, test, test_y)
    #     break

    # for season&playoff
    # for k,v in seasons_mode_blocks_dict.items():
    #     train, train_y, test, test_y = split_data(v[0], all_points, 'mode')
    #     print(train, train_y, test, test_y)
    #     break


    # train, train_y, test, test_y = split_data(raw, all_points)
    # best_n, best_m = find_RandomForest_parameters(train)
    # pred_train, pred_test = run_RandomForest(train, train_y, test, best_n, best_m)

    # pred_train, pred_test = run_LogisticRegression(train, train_y, test)

    # pred_train, pred_test = run_NaiveBayes(train, train_y, test)


    # print(train)
    # predictions_result(pred_train, train_y, pred_test, test_y)
'''


raw, unsorted_raw = process_data()
seasons_dict, seasons_loc_dict, seasons_mode_dict = split_for_experiments(raw)
seasons_blocks_dict = split_in_blocks(seasons_dict, 'season')
all_points = get_all_testing_points(unsorted_raw)

# print(seasons_blocks_dict)
