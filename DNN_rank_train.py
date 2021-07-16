import pandas as pd
import numpy as np
import tensorflow as tf
from data import *
from models import *
import time
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


DIR_NAME = PRO_DATA_TEST_DIR
MODULE_FILE = 'prediction.model'
UPDATE_FILE = 'updates_del'
INPUT_SIZE = len(FEATURES)
OUTPUT_SIZE = 3
HIDDEN_SIZE = INPUT_SIZE * 2 # GCN hidden size
BATCH_SIZE = NUM_HEROS_USED
TRAIN_END = 16  #train until the TRAIN_END th version, where TRAIN_END+NUM_PRED-1 < num_version
TRAIN_TIMES = 100
REGULARIZATION_RATE = 0.001  # L2 norm
LR = 0.003         # learning rate
NODELAMBDA = 0.2
RANKLAMBDA = 0.5
RAND_SEED = 99

COST_MATRIX_OVER = 3
COST_MATRIX_OPPOSITE = 5
COST_MATRIX = np.array([[0,1,1],[COST_MATRIX_OVER,0,COST_MATRIX_OPPOSITE],[COST_MATRIX_OVER,COST_MATRIX_OPPOSITE,0]])

#——————————————————load data——————————————————————
df_l = loadData(DIR_NAME, UPDATE_FILE)
label = df_l.iloc[:NUM_HEROS_USED,:].values #(114,35)
normalize_label = labelEncoding(label)    #(35,114,3)
versions = df_l.columns[:].values
for i in range(len(versions)):
    if versions[i] == '7':
        versions[i] = '7.00'
    elif versions[i] == '7.1':
        versions[i] = '7.10'


test_label = np.array(label[:,TRAIN_END:])
total_up, total_down = 0.0, 0.0
for i in range(len(test_label)):
    for j in range(len(test_label[0])):
        if test_label[i][j] == 1:
            total_up += 1
        elif test_label[i][j] == -1:
            total_down += 1





A, X_input = [], []
for i in range(len(versions)):
    adj, _, _, feature = load_data_lstmgcn(DIR_NAME, i, versions)
    # normalize_data = (feature-np.mean(feature,axis=0))/np.std(feature,axis=0) # (114,7)
    normalize_data = feature/np.mean(feature,axis=0) # (114,7)
    normalize_adj = adj
    normalize_adj[normalize_adj == 0] = 0.001
    A.append(normalize_adj)   # (35,114,114)
    X_input.append(normalize_data)    # (35,114,7)
X_input = np.array(X_input)

#——————————————————train and predict—————————————————————
def train_dnn():
    model = DNN_rank(INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, LR, REGULARIZATION_RATE, HIDDEN_SIZE, NODELAMBDA, RANKLAMBDA, COST_MATRIX)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # train
    for i in range(TRAIN_TIMES+1):
        start = 0
        while(start < TRAIN_END - 1):
            feed_dict = {
                    model.adj: A[start],
                    model.xs: X_input[start],
                    model.ys: normalize_label[start]
                    # initil state
            }
            _, cost, _,  = sess.run(
                [model.train_op, model.cost, model.pred, ],
                feed_dict=feed_dict)
            start += 1
        # if i % 100 == 0:
        #     print(i, 'cost:', cost)
    saver.save(sess, MODULE_FILE)

    # prediction
    #saver.restore(sess, MODULE_FILE)
    # the index th version for test。shape=[1,time_step,input_size]

    pred, hero_rank = sess.run([model.pred, model.output_rank],feed_dict={model.adj: A[TRAIN_END], model.xs:X_input[TRAIN_END]})

    test_predict = np.argmax(np.array(pred), axis=1)  # (114)
    test_y = np.argmax(normalize_label[TRAIN_END], axis=1)   # (114)

    #——————————————————Calc HitRatio——————————————————————
    print(versions[TRAIN_END], "--------------------", TRAIN_END)

    k_up = 50
    k_down = 30
    
    hero_rank_down, hero_rank_up = calcRank(np.array(hero_rank)[:,0], False)  # (114)
    hit_up, hit_down, F1_up, F1_down = calcHitRatio(hero_rank_down, test_y, k_up, k_down)

    hero_rank_down2, hero_rank_up2 = calcRank(np.array(hero_rank)[:,0], True)  # (114)
    hit_up2, hit_down2, F1_up2, F1_down2 = calcHitRatio(hero_rank_down2, test_y, k_up, k_down)


    if hit_down < hit_down2:
        hit_up = hit_up2
        hit_down = hit_down2
        F1_up = F1_up2
        F1_down = F1_down2
        hero_rank_down = hero_rank_down2
        hero_rank_up = hero_rank_up2

    count_up = np.zeros(len(hero_rank_down))
    count_down = np.zeros(len(hero_rank_down))
    for i in range(len(hero_rank_down)):
        if label[i,TRAIN_END] == 1:
            count_up[int(hero_rank_down[i])] = 1
        elif label[i,TRAIN_END] == -1:
            count_down[int(hero_rank_down[i])] = 1
    for i in range(1, len(hero_rank_down)):
        count_up[len(hero_rank_down) - i - 1] += count_up[len(hero_rank_down) - i]
        count_down[i] += count_down[i-1]
    count_up = count_up[::-1]

    df_m = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['measures'])
    for i in range(len(hero_rank_down)):
        df_m.loc[0, 'measures'] = hit_up
        df_m.loc[1, 'measures'] = hit_down
        df_m.loc[2 + i, 'measures'] = hero_rank_down[i]
        df_m.loc[2 + i + NUM_HEROS_USED, 'measures'] = count_up[i]
        df_m.loc[2 + i + NUM_HEROS_USED * 2, 'measures'] = count_down[i]
    return df_m


if __name__ == '__main__':
    start = time.time()
    Origin_train_end = TRAIN_END

    for seed in range(20):
        RAND_SEED = seed
        print("SEED~~~~~~~~~~~~!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", RAND_SEED)
        TRAIN_END = Origin_train_end

        df_measure = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['HitRatio'])
        for i in range(len(normalize_label) - TRAIN_END):
            tf.set_random_seed(RAND_SEED)
            df_m = train_dnn()
            df_measure[versions[TRAIN_END]] = df_m['measures']
            tf.compat.v1.reset_default_graph()
            TRAIN_END += 1
        tmp = np.array(df_measure.iloc[:,1:].values)
        for i in range(2 + NUM_HEROS_USED, 2 + NUM_HEROS_USED * 3):
            if i < 2 + NUM_HEROS_USED * 2:
                df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_up
            else:
                df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_down
        df_measure.loc[0, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 9, 'HitRatio']
        df_measure.loc[1, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 19, 'HitRatio']
        df_measure.loc[2, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 49, 'HitRatio']
        df_measure.loc[3, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 9, 'HitRatio']
        df_measure.loc[4, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 19, 'HitRatio']
        df_measure.loc[5, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 49, 'HitRatio']
        df_measure.to_csv(DIR_NAME + '/' + 'DNN_SEED_' + str(RAND_SEED) + '.csv')

    end = time.time()
    print("Time cost:", end - start)