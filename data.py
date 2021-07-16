'''
Created on 2019-07-24
@author: Han YUE
'''
import os
import json
import pandas as pd
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

NUM_HEROS = 130
NUM_HEROS_USED=112
NUM_PLAYERS_PER_GAME = 10
COST_MATRIX_UP = 3
COST_MATRIX_DOWN = 3
COST_MATRIX_OPPOSITE = 3
COST_MATRIX = np.array([[0,1,1],[COST_MATRIX_UP,0,COST_MATRIX_OPPOSITE],[COST_MATRIX_DOWN,COST_MATRIX_OPPOSITE,0]])
# TEST_DIR = "E:\\workspace\\Python\\GitHub\\stock_predict\\dataset"
# BACKUP_DIR = "E:\\workspace\\Python\\Dota2HeroChanges\\back_up\\pro_data"
# PRO_DATA_DIR = "E:\\workspace\\Python\\Dota2HeroChanges\\pro_data"
# AMATUER_DATA_DIR = "E:\\workspace\\Python\\Dota2HeroChanges\\amatuer_data"
# ALL_DATA_DIR = "E:\\workspace\\Python\\Dota2HeroChanges\\all_data"
# FEATURES = ['num_pick', 'num_ban', 'kda', 'total_gold', 'total_xp', 'hero_damage', 'tower_damage']
# PRO_DATA_TEST_DIR = '/home/dkeeper/Workspace/Dota2HeroChanges_Github/pro_data_test'
# PRO_DATA_DEL_DIR = '/home/dkeeper/Workspace/Dota2HeroChanges_Github/pro_data_delete_hero'
PRO_DATA_TEST_DIR = 'E:\\workspace\\Github\\Dota2HeroChanges\\pro_data_test'
PRO_DATA_DEL_DIR = 'E:\\workspace\\Github\\Dota2HeroChanges\\pro_data_delete_hero'
FEATURES = ['num_pick', 'num_ban', 'kda', 'total_gold', 'total_xp', 'hero_damage', 'tower_damage', 'win', 'kills', 'assists', 'deaths', 'duration', 'friend_score', 'opponent_score', 'radiant_gold_adv', 'radiant_xp_adv', 'hero_healing', 'neutral_kills', 'tower_kills', 'lane_kills', 'ancient_kills', 'obs_placed', 'sen_placed', 'rune_pickups', 'stuns', 'life_state_dead']
# FEATURES = ['teamfight_participation']
AVG_FEATURES = ['win']
FILTER=['.json']


def collectData(dirname, feature, is_rank=True):
    is_switch_hero = True
    df_l = loadData(PRO_DATA_TEST_DIR, 'updates')
    versions = df_l.columns[:].values
    for i in range(len(versions)):
        if versions[i] == '7':
            versions[i] = '7.00'
        elif versions[i] == '7.1':
            versions[i] = '7.10'

    # get folders
    dirs = []
    dir_list = os.listdir(dirname)
    for cur_file in dir_list:
        if os.path.isdir(os.path.join(dirname, cur_file)):
            dirs.append(cur_file)


    # df_main: all data
    df_main = pd.DataFrame(np.zeros((NUM_HEROS+1, len(versions))), columns=versions)

    # process data for each folder
    for cur_dir in dirs:
        vers = cur_dir
        cur_dir = os.path.join(dirname, cur_dir)
        print("%s" % cur_dir)
        cur_file_list = os.listdir(cur_dir)
        df_cur = pd.DataFrame(np.zeros((NUM_HEROS+1, 2)), columns=['hero_count', feature])
        df_graph = pd.DataFrame(np.zeros((NUM_HEROS, NUM_HEROS)))
        df_graph_herocount = pd.DataFrame(np.zeros((NUM_HEROS, NUM_HEROS)))
        df_graph_win = pd.DataFrame(np.zeros((NUM_HEROS, NUM_HEROS)))
        num_valid_file = 0
        invalid_files = []

        # process data fro each file
        for cur_file in cur_file_list:
            if os.path.splitext(cur_file)[1] in FILTER:
                file_path = os.path.join(cur_dir, cur_file)

                with open(file_path) as f_json:
                    load_dict = json.load(f_json)

                if load_dict['players'] == None:
                    print("No player info in this file: %s!" % file_path)
                    invalid_files.append(cur_file)
                    os.remove(file_path)
                elif load_dict['human_players'] != NUM_PLAYERS_PER_GAME or len(load_dict['players']) != NUM_PLAYERS_PER_GAME:
                    print("Not 10 Human players in this file: %s!" % file_path)
                    invalid_files.append(cur_file)
                    os.remove(file_path)
                elif load_dict['teamfights'] == None:
                    print("No teamfights info: %s!" % file_path)
                    invalid_files.append(cur_file)
                    # os.remove(file_path)
                # elif load_dict['picks_bans'] == None:
                #     print("No Picks_Bans Info in this file: %s!" % file_path)
                #     invalid_files.append(cur_file)
                #     os.remove(file_path)
                else:
                    file_is_valid = True
                    for i in range(NUM_PLAYERS_PER_GAME):
                        if load_dict['players'][i]['hero_id'] <= 0 or load_dict['players'][i]['hero_id']>NUM_HEROS:# or load_dict['players'][i]['hero_id']==105:
                            print("Hero_Id Error in this file: %s!" % file_path)
                            file_is_valid = False
                            invalid_files.append(cur_file)
                            os.remove(file_path)
                            break
                        elif load_dict['players'][i]['xp_per_min'] <= 0:
                            print("Player didn't play in this file: %s!" % file_path)
                            file_is_valid = False
                            invalid_files.append(cur_file)
                            os.remove(file_path)
                            break
                        # elif load_dict['players'][i]['teamfight_participation'] == None:
                        #     print("No teamfight_part in this file: %s!" % file_path)
                        #     file_is_valid = False
                        #     invalid_files.append(cur_file)
                        #     # os.remove(file_path)
                        #     break
                    if file_is_valid:
                        if feature == 'num_ban':
                            if load_dict['picks_bans'] == None:
                                #print("No Picks_Bans Info in this file: %s!" % file_path)
                                file_is_valid = False
                                invalid_files.append(cur_file)
                                #os.remove(file_path)
                            else:
                                for i in range(len(load_dict['picks_bans'])):
                                    if load_dict['picks_bans'][i]['is_pick'] == False:
                                        df_cur.loc[load_dict['picks_bans'][i]['hero_id']-1, 'num_ban'] += 1
                                    else:
                                        # df_main.loc[load_dict['picks_bans'][i]['hero_id']-1, 'hero_count'] += 1
                                        df_cur.loc[load_dict['picks_bans'][i]['hero_id']-1, 'hero_count'] += 1
                        elif feature == 'num_pick':
                            for i in range(NUM_PLAYERS_PER_GAME):
                                # df_main.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] += 1
                        elif feature == 'duration':
                            for i in range(NUM_PLAYERS_PER_GAME):
                                # df_main.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] += load_dict[feature]
                        elif feature == 'friend_score' or feature == 'opponent_score':
                            for i in range(NUM_PLAYERS_PER_GAME):
                                # df_main.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                if (load_dict['players'][i]['isRadiant'] == True and feature == 'friend_score') or (load_dict['players'][i]['isRadiant'] == False and feature == 'opponent_score'):
                                    df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] += load_dict['radiant_score']
                                else:
                                    df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] += load_dict['dire_score']
                        elif feature == 'radiant_gold_adv' or feature == 'radiant_xp_adv':
                            for i in range(NUM_PLAYERS_PER_GAME):
                                # df_main.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                if load_dict['players'][i]['isRadiant'] == True:
                                    df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] += load_dict[feature][-1]
                                else:
                                    df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] -= load_dict[feature][-1]
                        elif feature == 'kill_graph':
                            heroId = loadData(dirname, 'hero_id')
                            for i in range(NUM_PLAYERS_PER_GAME):
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                for j in range(NUM_PLAYERS_PER_GAME):
                                    df_graph_herocount.loc[load_dict['players'][i]['hero_id']-1, load_dict['players'][j]['hero_id']-1] += 1
                                    if load_dict['players'][i]['win'] == 1 and load_dict['players'][j]['win'] == 0:
                                        df_graph_win.loc[load_dict['players'][i]['hero_id']-1, load_dict['players'][j]['hero_id']-1] += 1
                                if len(load_dict['players'][i]['kills_log']) > 0:
                                    for j in range(len(load_dict['players'][i]['kills_log'])):
                                        df_graph.loc[load_dict['players'][i]['hero_id']-1, int(heroId.loc[0, load_dict['players'][i]['kills_log'][j]['key']])-1] += 1
                        elif feature in AVG_FEATURES or is_rank == False:
                            for i in range(NUM_PLAYERS_PER_GAME):
                                # df_main.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] += load_dict['players'][i][feature]
                        else:
                            for i in range(NUM_PLAYERS_PER_GAME):
                                cur_rank = -5.5
                                # df_main.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, 'hero_count'] += 1
                                for j in range(NUM_PLAYERS_PER_GAME):
                                    if load_dict['players'][i][feature] >= load_dict['players'][j][feature]:
                                        cur_rank += 1
                                df_cur.loc[load_dict['players'][i]['hero_id']-1, feature] += cur_rank                           
                    if file_is_valid:
                        num_valid_file += 1


        # normalization
        df_cur.loc[NUM_HEROS, feature] = num_valid_file
        if feature == 'num_pick' or feature == 'num_ban':
            for i in range(NUM_HEROS):
                df_cur.loc[i, feature] = df_cur.loc[i, feature]/num_valid_file
        elif feature == 'kill_graph':
            for i in range(NUM_HEROS):
                for j in range(NUM_HEROS):
                    if df_graph_herocount.iloc[i, j] != 0:
                            df_graph.iloc[i, j] = df_graph.iloc[i, j]/df_graph_herocount.iloc[i, j]
                            df_graph_win.iloc[i, j] = df_graph_win.iloc[i, j]/df_graph_herocount.iloc[i, j]
        else:
            for i in range(NUM_HEROS):
                if df_cur.loc[i, 'hero_count'] != 0:
                    df_cur.loc[i, feature] = df_cur.loc[i, feature]/df_cur.loc[i, 'hero_count']

        # save data of current version
        if feature == 'kill_graph':
            if is_switch_hero:
                df_graph.loc[23, :] = df_graph.loc[112, :]
                df_graph.loc[104, :] = df_graph.loc[113, :]
                df_graph.loc[:, 23] = df_graph.loc[:, 112]
                df_graph.loc[:, 104] = df_graph.loc[:, 113]
                df_graph_herocount.loc[23, :] = df_graph_herocount.loc[112, :]
                df_graph_herocount.loc[104, :] = df_graph_herocount.loc[113, :]
                df_graph_herocount.loc[:, 23] = df_graph_herocount.loc[:, 112]
                df_graph_herocount.loc[:, 104] = df_graph_herocount.loc[:, 113]
                df_graph_win.loc[23, :] = df_graph_herocount.loc[112, :]
                df_graph_win.loc[104, :] = df_graph_herocount.loc[113, :]
                df_graph_win.loc[:, 23] = df_graph_herocount.loc[:, 112]
                df_graph_win.loc[:, 104] = df_graph_herocount.loc[:, 113]
            df_graph.to_csv(cur_dir + '/' + feature + '.csv')
            df_graph_herocount.to_csv(cur_dir + '/' + feature + '_herocount.csv')
            df_graph_win.to_csv(cur_dir + '/' + feature + '_win.csv')
        else:
            if is_switch_hero:
                df_cur.loc[23, feature] = df_cur.loc[112, feature]
                df_cur.loc[104, feature] = df_cur.loc[113, feature]
                df_cur.loc[23, 'hero_count'] = df_cur.loc[112, 'hero_count']
                df_cur.loc[104, 'hero_count'] = df_cur.loc[113, 'hero_count']
            df_cur.to_csv(cur_dir + '/' + feature + '.csv')
            saver_path = cur_dir + '/valid_files.txt'
            with open(saver_path, 'a+') as f_saver:
                f_saver.write(str(num_valid_file))
                f_saver.write('\n')
                f_saver.write(str(invalid_files))
                f_saver.write('\n')
            df_main[vers] = df_cur[feature]

    # save data of all versions
    if feature != 'kill_graph':
        # if is_switch_hero:
        #     df_main.loc[23, 'hero_count'] = df_main.loc[112, 'hero_count']      #替换空英雄
        #     df_main.loc[104, 'hero_count'] = df_main.loc[113, 'hero_count']     #替换炸弹人
        df_main.to_csv(dirname + '/' + feature + '.csv')


# read data, indexed by hero id
def loadData(dirname, feature):
    file_path = dirname + '/' + feature + '.csv'
    with open(file_path) as f_csv:
        df=pd.read_csv(f_csv)
    return df


# for feature in FEATURES:
#     print("%s start --------------------------------------------------" % feature)
#     collectData(PRO_DATA_TEST_DIR, feature, False)
#     print("%s end ----------------------------------------------------" % feature)
# collectData(PRO_DATA_TEST_DIR, 'teamfight_participation')
# collectData(PRO_DATA_DEL_DIR, 'kill_graph')



def load_data_lstmgcn(dirname, versionIndex, versions): # read data
    df_features = pd.DataFrame(np.zeros((NUM_HEROS_USED, len(FEATURES))), columns=FEATURES)
    for feature in FEATURES:
        df_cur = loadData(dirname, feature)
        df_features[feature] = df_cur.iloc[:NUM_HEROS_USED, versionIndex + 1]   # first line: id-1
    features = df_features.iloc[:NUM_HEROS_USED, :].values

    version_path = dirname + '/' + versions[versionIndex]
    adj = loadData(version_path, 'kill_graph').iloc[:NUM_HEROS_USED, 1:NUM_HEROS_USED+1].values
    adj_count = loadData(version_path, 'kill_graph_herocount').iloc[:NUM_HEROS_USED, 1:NUM_HEROS_USED+1].values
    adj_win = loadData(version_path, 'kill_graph_win').iloc[:NUM_HEROS_USED, 1:NUM_HEROS_USED+1].values
    return adj, adj_count, adj_win, features


# df_l = loadData(PRO_DATA_TEST_DIR, 'updates')
# vers = df_l.columns[:].values
# for i in range(len(vers)):
#     if vers[i] == '7':
#         vers[i] = '7.00'
#     elif vers[i] == '7.1':
#         vers[i] = '7.10'
# load_data_lstmgcn(PRO_DATA_TEST_DIR, 0, vers)



# one-hot labelling
def labelEncoding(label):
    batch = label.shape[0]    # 114
    steps = label.shape[1]    # 35
    output = []
    for i in range(steps):
        cur_hero = []   # (114,3)
        for j in range(batch):
            if label[j][i] == 1:
                cur_hero.append([0, 1, 0])  # enhanced
            elif label[j][i] == -1:
                cur_hero.append([0, 0, 1])  # weakened
            else:
                cur_hero.append([1, 0, 0])  # unchanged
        output.append(np.array(cur_hero))
    return np.array(output)    # (35,114,3)



# calculate F1 score
def CalcF1score(test_predict, test_y, NUM_PRED):
    result, Ac, Pr, Re, F1 = [], [], [], [], []
    result_Up, Ac_Up, Pr_Up, Re_Up, F1_Up = [], [], [], [], []
    result_Down, Ac_Down, Pr_Down, Re_Down, F1_Down = [], [], [], [], []
    df_m = pd.DataFrame(np.zeros((NUM_PRED*7 + NUM_HEROS_USED, 1)), columns=['measures'])
    for i in range(len(test_y)):
        predUpdate, labelUpdate, rightUpdate, rightUnchanged, accuracy, precision, recall, f1score = 0, 0, 0, 0, 0, 0, 0, 0
        pred_Up, label_Up, right_Up, accuracy_Up, precision_Up, recall_Up, f1score_Up = 0, 0, 0, 0, 0, 0, 0
        pred_Down, label_Down, right_Down, accuracy_Down, precision_Down, recall_Down, f1score_Down = 0, 0, 0, 0, 0, 0, 0
        for j in range(NUM_HEROS_USED):
            if test_predict[i][j] == test_y[i][j]:
                if test_predict[i][j] == 1:
                    pred_Up += 1
                    label_Up += 1
                    right_Up += 1
                elif test_predict[i][j] == 2:
                    pred_Down += 1
                    label_Down += 1
                    right_Down += 1
                else:
                    rightUnchanged += 1
            else:
                if test_predict[i][j] == 1:
                    pred_Up += 1
                elif test_predict[i][j] == 2:
                    pred_Down += 1
                if test_y[i][j] == 1:
                    label_Up += 1
                elif test_y[i][j] == 2:
                    label_Down += 1

        predUpdate = pred_Up + pred_Down
        labelUpdate = label_Up + label_Down
        rightUpdate = right_Up + right_Down

        TP = rightUpdate
        TN = rightUnchanged
        FP = predUpdate - rightUpdate
        FN = labelUpdate - rightUpdate
        accuracy = (TP + TN) / NUM_HEROS_USED
        if predUpdate != 0:
            precision = TP/(TP + FP)
        else:
            precision = 0
        recall = TP/(TP + FN)
        if (precision + recall) != 0:
            f1score = 2 * precision * recall / (precision + recall)
        else:
            f1score = 0
        
        TP_Up = right_Up
        TP_Down = right_Down
        TN_Up = rightUnchanged + right_Down
        TN_Down = rightUnchanged + right_Up
        FP_Up = pred_Up - right_Up
        FP_Down = pred_Down - right_Down
        FN_Up = label_Up - right_Up
        FN_Down = label_Down - right_Down
        accuracy_Up = (TP_Up + TN_Up) / NUM_HEROS_USED
        accuracy_Down = (TP_Down + TN_Down) / NUM_HEROS_USED
        if pred_Up != 0:
            precision_Up = TP_Up/(TP_Up + FP_Up)
        else:
            precision_Up = 0
        if pred_Down != 0:
            precision_Down = TP_Down/(TP_Down + FP_Down)
        else:
            precision_Down = 0
        if (TP_Up + FN_Up) != 0:
            recall_Up = TP_Up/(TP_Up + FN_Up)
        else:
            recall_Up = 0
        if (TP_Down + FN_Down) != 0:
            recall_Down = TP_Down/(TP_Down + FN_Down)
        else:
            recall_Down = 0
        if (precision_Up + recall_Up) != 0:
            f1score_Up = 2 * precision_Up * recall_Up / (precision_Up + recall_Up)
        else:
            f1score_Up = 0
        if (precision_Down + recall_Down) != 0:
            f1score_Down = 2 * precision_Down * recall_Down / (precision_Down + recall_Down)
        else:
            f1score_Down = 0

        result.append([predUpdate, labelUpdate, rightUpdate])
        Ac.append(accuracy)
        Pr.append(precision)
        Re.append(recall)
        F1.append(f1score)

        result_Up.append([pred_Up, label_Up, right_Up])
        Ac_Up.append(accuracy_Up)
        Pr_Up.append(precision_Up)
        Re_Up.append(recall_Up)
        F1_Up.append(f1score_Up)
        result_Down.append([pred_Down, label_Down, right_Down])
        Ac_Down.append(accuracy_Down)
        Pr_Down.append(precision_Down)
        Re_Down.append(recall_Down)
        F1_Down.append(f1score_Down)
    for i in range(len(Pr)):
        df_m.loc[i, 'measures'] = F1[i]
    for i in range(len(Re)):
        df_m.loc[i+NUM_PRED, 'measures'] = F1_Up[i]
    for i in range(len(F1)):
        df_m.loc[i+NUM_PRED*2, 'measures'] = F1_Down[i]
    if NUM_PRED == 1:
        df_m.loc[3, 'measures'] = pred_Up
        df_m.loc[4, 'measures'] = right_Up
        df_m.loc[5, 'measures'] = pred_Down
        df_m.loc[6, 'measures'] = right_Down
        for i in range(NUM_HEROS_USED):
            df_m.loc[7+i, 'measures'] = test_predict[0][i]


    # print("result[predUpdate, labelUpdate, rightUpdate]-------------------------")
    print("Both--", result, F1)
    print("UP----", result_Up, F1_Up)
    print("Down--", result_Down, F1_Down)
    # print(df_m)

    return df_m




# calculte F1 score for non-LSTM model
def CalcF1score_NoLSTM(test_predict, test_y, up, donw):
    result, result_Up, result_Down = [], [], []
    df_m = pd.DataFrame(np.zeros((7+NUM_HEROS_USED, 1)), columns=['measures'])

    predUpdate, labelUpdate, rightUpdate, rightUnchanged, precision, recall, f1score = 0, 0, 0, 0, 0, 0, 0
    pred_Up, label_Up, right_Up, precision_Up, recall_Up, f1score_Up = 0, 0, 0, 0, 0, 0
    pred_Down, label_Down, right_Down, precision_Down, recall_Down, f1score_Down = 0, 0, 0, 0, 0, 0
    for j in range(NUM_HEROS_USED):
        if test_predict[j] == test_y[j]:
            if test_predict[j] == up:
                pred_Up += 1
                label_Up += 1
                right_Up += 1
            elif test_predict[j] == donw:
                pred_Down += 1
                label_Down += 1
                right_Down += 1
            else:
                rightUnchanged += 1
        else:
            if test_predict[j] == up:
                pred_Up += 1
            elif test_predict[j] == donw:
                pred_Down += 1
            if test_y[j] == up:
                label_Up += 1
            elif test_y[j] == donw:
                label_Down += 1

    predUpdate = pred_Up + pred_Down
    labelUpdate = label_Up + label_Down
    rightUpdate = right_Up + right_Down

    TP = rightUpdate
    FP = predUpdate - rightUpdate
    FN = labelUpdate - rightUpdate
    if predUpdate != 0:
        precision = TP/(TP + FP)
    else:
        precision = 0
    recall = TP/(TP + FN)
    if (precision + recall) != 0:
        f1score = 2 * precision * recall / (precision + recall)
    else:
        f1score = 0
    
    TP_Up = right_Up
    TP_Down = right_Down
    FP_Up = pred_Up - right_Up
    FP_Down = pred_Down - right_Down
    FN_Up = label_Up - right_Up
    FN_Down = label_Down - right_Down
    if pred_Up != 0:
        precision_Up = TP_Up/(TP_Up + FP_Up)
    else:
        precision_Up = 0
    if pred_Down != 0:
        precision_Down = TP_Down/(TP_Down + FP_Down)
    else:
        precision_Down = 0
    if (TP_Up + FN_Up) != 0:
        recall_Up = TP_Up/(TP_Up + FN_Up)
    else:
        recall_Up = 0
    if (TP_Down + FN_Down) != 0:
        recall_Down = TP_Down/(TP_Down + FN_Down)
    else:
        recall_Down = 0
    if (precision_Up + recall_Up) != 0:
        f1score_Up = 2 * precision_Up * recall_Up / (precision_Up + recall_Up)
    else:
        f1score_Up = 0
    if (precision_Down + recall_Down) != 0:
        f1score_Down = 2 * precision_Down * recall_Down / (precision_Down + recall_Down)
    else:
        f1score_Down = 0

    result.append([predUpdate, labelUpdate, rightUpdate, rightUnchanged])
    result_Up.append([pred_Up, label_Up, right_Up, rightUnchanged+right_Down])
    result_Down.append([pred_Down, label_Down, right_Down, rightUnchanged+right_Up])

    df_m.loc[0, 'measures'] = f1score
    df_m.loc[1, 'measures'] = f1score_Up
    df_m.loc[2, 'measures'] = f1score_Down
    df_m.loc[3, 'measures'] = pred_Up
    df_m.loc[4, 'measures'] = right_Up
    df_m.loc[5, 'measures'] = pred_Down
    df_m.loc[6, 'measures'] = right_Down
    for i in range(NUM_HEROS_USED):
        df_m.loc[7+i, 'measures'] = test_predict[i]

    # print("result[predUpdate, labelUpdate, rightUpdate, rightUnchanged]-------------------------")
    print("Both--", result, f1score)
    print("UP----", result_Up, f1score_Up)
    print("Down--", result_Down, f1score_Down)

    # print(TRAIN_END, versions[TRAIN_END], "------" , df_m)
    return df_m




# calc reciprocal for matrix, A:N*N
def calcReciprocal(A):
    num_nodes_1 = A.shape[0]
    num_nodes_2 = A.shape[1]
    B = np.zeros([num_nodes_1, num_nodes_2])
    for j in range(num_nodes_1):
        for k in range(num_nodes_2):
            if A[j][k] != 0:
                B[j][k] = 1/A[j][k]
    return B


# calc laplacian matrix, A:N*N
def calculate_laplacian(A):
    adj = A + np.eye(A.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.matrix(np.diag(d_inv_sqrt))
    L = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return L


# calc pagerank, A:N*N
def calcPageRank(A, dampingFactor):
    num_nodes = len(A)
    Origin_PR = np.ones(num_nodes)/num_nodes
    PR = np.ones(num_nodes)
    for i in range(num_nodes):
        PR[i] = (1 - dampingFactor)/num_nodes
        for j in range(num_nodes):
            out = A[j].sum(0)
            if out != 0:
                PR[i] += A[j][i] * dampingFactor * Origin_PR[j] / out
    return PR


# replace 0s by average
def zero2mean(Matrix, allFill=False):
    m = Matrix.shape[0]
    n = Matrix.shape[1]
    result = Matrix.copy()
    for j in range(n):
        count, avg = 0, 0
        for i in range(m):
            if Matrix[i][j] != 0:
                count += 1
                avg += Matrix[i][j]
        if count != 0:
            avg = avg/count
        for i in range(m):
            if result[i][j] == 0:
                result[i][j] = avg
    if allFill == True:
        for i in range(m):
            count, avg = 0, 0
            for j in range(n):
                if Matrix[i][j] != 0:
                    count += 1
                    avg += Matrix[i][j]
            if count != 0:
                avg = avg/count
            for j in range(n):
                if result[i][j] == 0:
                    result[i][j] = avg
    return result


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


# return ranking info of an array
def calcRank(arr, isReverse):
    result_down = np.zeros(len(arr))
    result_up = np.ones(len(arr)) * len(arr)
    srt = sorted(arr, reverse=isReverse)
    for i in range(len(arr)):
        for j in range(len(srt)):
            if srt[j] == arr[i]:
                break
        result_down[i] = j
        for k in range(len(srt)):
            if srt[len(arr)-1-k] == arr[i]:
                break
        result_up[i] = len(arr)-1-k

    return result_down, result_up


# Get number of changes from encoded labels (35,114,3)
def labelSum(label):
    return np.sum(label, axis=1)    # (35,114,3)


def calcHitRatio(arr_rank, arr_label, k_up, k_down):
    cur_rank_label = np.zeros(len(arr_rank))

    for i in range(len(arr_rank)):
        if arr_rank[i] >= (len(arr_rank) - k_up):
            cur_rank_label[i] = 1
        elif arr_rank[i] < k_down:
            cur_rank_label[i] = 2

    right_up, right_down, label_up, label_down, pred_up, pred_down = 0, 0, 0, 0, 0, 0

    for i in range(len(arr_rank)):
        if cur_rank_label[i] == arr_label[i]:
            if arr_label[i] == 1:
                right_up += 1
                label_up += 1
                pred_up += 1
            elif arr_label[i] == 2:
                right_down += 1
                label_down += 1
                pred_down += 1
        else:
            if arr_label[i] == 1:
                label_up += 1
            elif arr_label[i] == 2:
                label_down += 1
            if cur_rank_label[i] == 1:
                pred_up += 1
            elif cur_rank_label[i] == 2:
                pred_down += 1
    
    if pred_up == 0:
        precision_up = 0
    else:
        precision_up = right_up/pred_up
    if pred_down == 0:
        precision_down = 0
    else:
        precision_down = right_down/pred_down

    if label_up == 0:
        hit_up = 0
        F1_up = 0
    else:
        hit_up = right_up/label_up
        if right_up == 0:
            F1_up = 0
        else:
            F1_up = 2 * hit_up * precision_up / (hit_up + precision_up)
    if label_down == 0:
        hit_down = 0
        F1_down = 0
    else:
        hit_down = right_down/label_down
        if right_down == 0:
            F1_down = 0
        else:
            F1_down = 2 * hit_down * precision_down / (hit_down + precision_down)

    # print(" ", k_up, "up--", round(hit_up,4), " ", k_down, "down--", round(hit_down,4))
    return hit_up, hit_down, F1_up, F1_down





def calcHitRatioBatch(arr_rank, arr_label, k_up, k_down):
    cur_rank_label = np.zeros(len(arr_rank))

    for i in range(len(arr_rank)):
        if arr_rank[i] >= (len(arr_rank) - k_up):
            cur_rank_label[i] = 1
        elif arr_rank[i] < k_down:
            cur_rank_label[i] = 2

    right_up, right_down, label_up, label_down, pred_up, pred_down = 0, 0, 0, 0, 0, 0

    for i in range(len(arr_rank)):
        if cur_rank_label[i] == arr_label[i]:
            if arr_label[i] == 1:
                right_up += 1
                label_up += 1
                pred_up += 1
            elif arr_label[i] == 2:
                right_down += 1
                label_down += 1
                pred_down += 1
        else:
            if arr_label[i] == 1:
                label_up += 1
            elif arr_label[i] == 2:
                label_down += 1
            if cur_rank_label[i] == 1:
                pred_up += 1
            elif cur_rank_label[i] == 2:
                pred_down += 1
    
    if pred_up == 0:
        precision_up = 0
    else:
        precision_up = right_up/pred_up
    if pred_down == 0:
        precision_down = 0
    else:
        precision_down = right_down/pred_down

    if label_up == 0:
        hit_up = 0
        F1_up = 0
    else:
        hit_up = right_up/label_up
        if right_up == 0:
            F1_up = 0
        else:
            F1_up = 2 * hit_up * precision_up / (hit_up + precision_up)
    if label_down == 0:
        hit_down = 0
        F1_down = 0
    else:
        hit_down = right_down/label_down
        if right_down == 0:
            F1_down = 0
        else:
            F1_down = 2 * hit_down * precision_down / (hit_down + precision_down)

    # print(" ", k_up, "up--", round(hit_up,4), " ", k_down, "down--", round(hit_down,4))
    return hit_down, right_up, right_down, label_up, label_down, pred_up, pred_down



def calcHitRatioBatch2(arr_r, arr_l, k_up, k_down):
    r_up, r_down, l_up, l_down, p_up, p_down = [], [], [], [], [], []
    for n in range(len(arr_r)):
        arr_rank = arr_r[n,:,1]
        arr_label = arr_l[n]
        cur_rank_label = np.zeros(len(arr_rank))

        for i in range(len(arr_rank)):
            if arr_rank[i] >= (len(arr_rank) - k_up):
                cur_rank_label[i] = 1
            elif arr_rank[i] < k_down:
                cur_rank_label[i] = 2

        right_up, right_down, label_up, label_down, pred_up, pred_down = 0, 0, 0, 0, 0, 0

        for i in range(len(arr_rank)):
            if cur_rank_label[i] == arr_label[i]:
                if arr_label[i] == 1:
                    right_up += 1
                    label_up += 1
                    pred_up += 1
                elif arr_label[i] == 2:
                    right_down += 1
                    label_down += 1
                    pred_down += 1
            else:
                if arr_label[i] == 1:
                    label_up += 1
                elif arr_label[i] == 2:
                    label_down += 1
                if cur_rank_label[i] == 1:
                    pred_up += 1
                elif cur_rank_label[i] == 2:
                    pred_down += 1
        
        if pred_up == 0:
            precision_up = 0
        else:
            precision_up = right_up/pred_up
        if pred_down == 0:
            precision_down = 0
        else:
            precision_down = right_down/pred_down

        if label_up == 0:
            hit_up = 0
            F1_up = 0
        else:
            hit_up = right_up/label_up
            if right_up == 0:
                F1_up = 0
            else:
                F1_up = 2 * hit_up * precision_up / (hit_up + precision_up)
        if label_down == 0:
            hit_down = 0
            F1_down = 0
        else:
            hit_down = right_down/label_down
            if right_down == 0:
                F1_down = 0
            else:
                F1_down = 2 * hit_down * precision_down / (hit_down + precision_down)

        # print(" ", k_up, "up--", round(hit_up,4), " ", k_down, "down--", round(hit_down,4))
        r_up.append(right_up)
        r_down.append(right_down)
        l_up.append(label_up)
        l_down.append(label_down)
        p_up.append(pred_up)
        p_down.append(pred_down)
    right_up = sum(r_up)
    right_down = sum(r_down)
    label_up = sum(l_up)
    label_down = sum(l_down)
    pred_up = sum(p_up)
    pred_down = sum(p_down)
    hit_up = right_up/label_up
    hit_down = right_down/label_down
    precision_up = right_up/pred_up
    precision_down = right_down/pred_down
    F1_up = 2 * hit_up * precision_up / (hit_up + precision_up)
    F1_down = 2 * hit_down * precision_down / (hit_down + precision_down)
    # print(" avg", k_up, "up--", round(hit_up,4), " ", k_down, "down--", round(hit_down,4))
    return hit_up, hit_down, F1_up, F1_down