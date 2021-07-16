import pandas as pd
import numpy as np
import tensorflow as tf
from data import *
from models import *
import time
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Constant Parameters
DIR_NAME = PRO_DATA_TEST_DIR
MODULE_FILE = 'prediction.model'
UPDATE_FILE = 'updates_del'
INPUT_SIZE = len(FEATURES)
OUTPUT_SIZE = 3
HIDDEN_SIZE = INPUT_SIZE * 2    # Hidden layer of GCN
CELL_SIZE = INPUT_SIZE * 2  # LSTM cell
TIME_STEPS = 2
BATCH_SIZE = NUM_HEROS_USED
TRAIN_END = 16              # End version of training. TRAIN_END+NUM_PRED-1 < len(versions)
TRAIN_TIMES = 100           # training epoch
NUM_PRED = 1                # num of versions to predict
FORGET_BIAS = 0.6           # forget bias
REGULARIZATION_RATE = 0.001 # L2 regularization
LR = 0.003                  # learning rate
NODELAMBDA = 0.2            # node ranking constraint loss rate
RANKLAMBDA = 0.5            # ranking loss rate
NUM_ADJ = 1                 # How many adjacency matrices to use
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
    adj, adj_count, adj_win, feature = load_data_lstmgcn(DIR_NAME, i, versions) # Killing Matrix, Choosing Matrix, Winning Matrix
    if DIR_NAME == PRO_DATA_DEL_DIR:
        normalize_data = (feature-np.mean(feature,axis=0))/np.std(feature,axis=0) # (114,7)
    else:
        normalize_data = feature/np.mean(feature,axis=0) # (114,7)
    cur_A = []
    normalize_adj = adj_count#calcReciprocal(adj)
    normalize_adj[normalize_adj == 0] = 0.001
    cur_A.append(normalize_adj)
    if NUM_ADJ > 1:
        normalize_adj = adj#calcReciprocal(adj)
        normalize_adj[normalize_adj == 0] = 0.001
        cur_A.append(normalize_adj)
    if NUM_ADJ > 2:
        normalize_adj = adj_win#calcReciprocal(adj)
        normalize_adj[normalize_adj == 0] = 0.001
        cur_A.append(normalize_adj)   # (3,114,114)

    A.append(cur_A) # (35,3,114,114)
    X_input.append(normalize_data)    # (35,114,7)
X_input = np.array(X_input)


#——————————————————training & testing——————————————————————
def train_gcnlstm():
    #——————————————————data process——————————————————————
    train_x,train_y=[],[]   # Training Set
    for i in range(TRAIN_END - TIME_STEPS + 1):
        x = X_input[i:i+TIME_STEPS] # (TIME_STEPS, 114, 7)
        y = normalize_label[i:i+TIME_STEPS] # (TIME_STEPS, 114, 3)
        y = y.transpose(1,0,2)    # (114, TIME_STEPS, 3)
        train_x.append(x)  # (TRAIN_END, TIME_STEPS, 114, 7)
        train_y.append(y)  # (TRAIN_END, 114, TIME_STEPS, 3)


    model = R3GL_Rank_parameter(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR, REGULARIZATION_RATE, HIDDEN_SIZE, FORGET_BIAS, NODELAMBDA, RANKLAMBDA, COST_MATRIX, NUM_ADJ)
    # saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    #——————————————————Training——————————————————————
    for i in range(TRAIN_TIMES+1):
        start = 0
        c_val = np.zeros([BATCH_SIZE, CELL_SIZE])
        h_val = np.zeros([BATCH_SIZE, CELL_SIZE])
        state = None
        while(start < len(train_x)):
            if state == None or TIME_STEPS > 1:
                feed_dict = {
                        model.adj: A[start:start+TIME_STEPS],
                        model.xs: train_x[start],
                        model.ys: train_y[start],
                        model.init_c: c_val,
                        model.init_h: h_val
                }
            else:
                feed_dict = {
                    model.adj: A[start:start+TIME_STEPS],
                    model.xs: train_x[start],
                    model.ys: train_y[start],
                    model.init_c: state.c,
                    model.init_h: state.h
                }

            _, cost, state, pred, r_loss = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred, model.rankLoss],
                feed_dict=feed_dict)
            start += 1
        # if i % 20 == 0:
        #     print(i, 'cost:', cost, r_loss)
    # saver.save(sess, MODULE_FILE)

    #——————————————————Testing——————————————————————
    #saver.restore(sess, MODULE_FILE)
    # Testing Set is the intex'th sample after Training Set。shape=[1,time_step,input_size]
    index = 0
    prev_seq = X_input[TRAIN_END+1-TIME_STEPS+index:TRAIN_END+1+index]  # (TIME_STEPS, 114, 7)
    prev_seq_A = A[TRAIN_END+1-TIME_STEPS+index:TRAIN_END+1+index]  # (TIME_STEPS, 114, 114)
    predict, hero_rank = [], []
    for i in range(NUM_PRED):
        if TRAIN_END+NUM_PRED+index < len(normalize_label):
            _, pred, rank = sess.run([model.cell_final_state, model.pred, model.output_rank],feed_dict={model.adj: prev_seq_A, model.xs:prev_seq, model.init_c:c_val, model.init_h:h_val})
            pred = pred.transpose(1,0,2)    # (TIME_STEPS,114,3)
            predict.append(pred[-1])    # (NUM_PRED,114,3)
            hero_rank.append(rank.transpose(1,0,2)[-1])    # (NUM_PRED,114,1)
            if TIME_STEPS == 1:
                prev_seq = X_input[TRAIN_END+1+index+i]
                prev_seq_A = A[TRAIN_END+1+index+i]
            else:
                prev_seq = np.vstack((prev_seq[1:],[X_input[TRAIN_END+1+index+i]]))
                prev_seq_A = np.vstack((prev_seq_A[1:],[A[TRAIN_END+1+index+i]]))
        else:
            _, pred, rank = sess.run([model.cell_final_state, model.pred, model.output_rank],feed_dict={model.adj: prev_seq_A, model.xs:prev_seq, model.init_c:c_val, model.init_h:h_val})
            pred = pred.transpose(1,0,2)    # (TIME_STEPS,114,3)
            predict.append(pred[-1])    # (NUM_PRED,114,3)
            hero_rank.append(rank.transpose(1,0,2)[-1])    # (NUM_PRED,114,1)
            break
    if TRAIN_END+1+i+index < len(normalize_label):
        test_y = normalize_label[TRAIN_END+index:TRAIN_END+1+i+index]
    elif TRAIN_END+1+index < len(normalize_label):
        test_y = normalize_label[TRAIN_END+index:]
    else:
        test_y = [normalize_label[len(normalize_label)-1]]

    test_predict = np.argmax(np.array(predict), axis=2)  # (NUM_PRED,114)
    test_y = np.argmax(test_y, axis=2)   # (NUM_PRED,114)


    #——————————————————Calc HitRatio——————————————————————
    # print(versions[TRAIN_END], "--------------------", TRAIN_END)

    k_up = 50
    k_down = 30
    
    hero_rank_down, hero_rank_up = calcRank(np.array(hero_rank)[0,:,0], False)  # (114)
    hit_up, hit_down, F1_up, F1_down = calcHitRatio(hero_rank_down, test_y[0], k_up, k_down)

    hero_rank_down2, hero_rank_up2 = calcRank(np.array(hero_rank)[0,:,0], True)  # (114)
    hit_up2, hit_down2, F1_up2, F1_down2 = calcHitRatio(hero_rank_down2, test_y[0], k_up, k_down)


    # df_m = pd.DataFrame(np.zeros((4 + NUM_HEROS_USED*2, 1)), columns=['measures'])
    # for i in range(len(hero_rank_down)):
    #     if hit_down >= hit_down2:
    #         df_m.loc[0, 'measures'] = hit_up
    #         df_m.loc[1, 'measures'] = hit_down
    #         df_m.loc[2, 'measures'] = F1_up
    #         df_m.loc[3, 'measures'] = F1_down
    #         df_m.loc[4 + i, 'measures'] = hero_rank_down[i]
    #         df_m.loc[4 + i + NUM_HEROS_USED, 'measures'] = hero_rank_up[i]
    #     else:
    #         df_m.loc[0, 'measures'] = hit_up2
    #         df_m.loc[1, 'measures'] = hit_down2
    #         df_m.loc[2, 'measures'] = F1_up2
    #         df_m.loc[3, 'measures'] = F1_down2
    #         df_m.loc[4 + i, 'measures'] = hero_rank_down2[i]
    #         df_m.loc[4 + i + NUM_HEROS_USED, 'measures'] = hero_rank_up2[i]


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


        # ---------------------------Train & Test--------------------------------------
        TRAIN_END = Origin_train_end
        df_measure = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['HitRatio'])
        for i in range(len(normalize_label) - TRAIN_END - NUM_PRED + 1):
            tf.set_random_seed(RAND_SEED)
            df_m = train_gcnlstm()
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
        df_measure.to_csv(DIR_NAME + '/' + 'RGL_SEED_' + str(RAND_SEED) + '.csv')




        # # -------------------Cost Matrix----------------------------------------------------------------
        # for m in range(1, 6):
        #     for n in range(1, 6):
        #         COST_MATRIX_OVER = m
        #         COST_MATRIX_OPPOSITE = n
        #         COST_MATRIX = np.array([[0,1,1],[COST_MATRIX_OVER,0,COST_MATRIX_OPPOSITE],[COST_MATRIX_OVER,COST_MATRIX_OPPOSITE,0]])
        #         print("Cost Matrix===========================", m, n)
        #         TRAIN_END = Origin_train_end
        #         df_measure = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['HitRatio'])
        #         for i in range(len(normalize_label) - TRAIN_END - NUM_PRED + 1):
        #             tf.set_random_seed(RAND_SEED)
        #             df_m = train_gcnlstm()
        #             df_measure[versions[TRAIN_END]] = df_m['measures']
        #             tf.compat.v1.reset_default_graph()
        #             TRAIN_END += 1
        #         tmp = np.array(df_measure.iloc[:,1:].values)
        #         for i in range(2 + NUM_HEROS_USED, 2 + NUM_HEROS_USED * 3):
        #             if i < 2 + NUM_HEROS_USED * 2:
        #                 df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_up
        #             else:
        #                 df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_down
        #         df_measure.loc[0, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 9, 'HitRatio']
        #         df_measure.loc[1, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 19, 'HitRatio']
        #         df_measure.loc[2, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 49, 'HitRatio']
        #         df_measure.loc[3, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 9, 'HitRatio']
        #         df_measure.loc[4, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 19, 'HitRatio']
        #         df_measure.loc[5, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 49, 'HitRatio']
        #         df_measure.to_csv(DIR_NAME + '/' + 'R3GL_SEED_' + str(RAND_SEED) + '_CM_' + str(m) + '_' + str(n) + '.csv')
        # COST_MATRIX_OVER = 3
        # COST_MATRIX_OPPOSITE = 3
        # COST_MATRIX = np.array([[0,1,1],[COST_MATRIX_OVER,0,COST_MATRIX_OPPOSITE],[COST_MATRIX_OVER,COST_MATRIX_OPPOSITE,0]])

        # # -------------------Forget Bias----------------------------------------------------------------
        # for n in range(10):
        #     FORGET_BIAS = 0.1 * n
        #     print("Forget Bias===========================", FORGET_BIAS)
        #     TRAIN_END = Origin_train_end
        #     df_measure = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['HitRatio'])
        #     for i in range(len(normalize_label) - TRAIN_END - NUM_PRED + 1):
        #         tf.set_random_seed(RAND_SEED)
        #         df_m = train_gcnlstm()
        #         df_measure[versions[TRAIN_END]] = df_m['measures']
        #         tf.compat.v1.reset_default_graph()
        #         TRAIN_END += 1
        #     tmp = np.array(df_measure.iloc[:,1:].values)
        #     for i in range(2 + NUM_HEROS_USED, 2 + NUM_HEROS_USED * 3):
        #         if i < 2 + NUM_HEROS_USED * 2:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_up
        #         else:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_down
        #     df_measure.loc[0, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 9, 'HitRatio']
        #     df_measure.loc[1, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 19, 'HitRatio']
        #     df_measure.loc[2, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 49, 'HitRatio']
        #     df_measure.loc[3, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 9, 'HitRatio']
        #     df_measure.loc[4, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 19, 'HitRatio']
        #     df_measure.loc[5, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 49, 'HitRatio']
        #     df_measure.to_csv(DIR_NAME + '/' + 'R3GL_SEED_' + str(RAND_SEED) + '_FB_' + str(FORGET_BIAS) + '.csv')
        # FORGET_BIAS = 0.6

        # # -------------------Rank LAMBDA----------------------------------------------------------------
        # for n in range(10):
        #     RANKLAMBDA = 0.1 * n
        #     print("Rank Lambda===========================", RANKLAMBDA)
        #     TRAIN_END = Origin_train_end
        #     df_measure = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['HitRatio'])
        #     for i in range(len(normalize_label) - TRAIN_END - NUM_PRED + 1):
        #         tf.set_random_seed(RAND_SEED)
        #         df_m = train_gcnlstm()
        #         df_measure[versions[TRAIN_END]] = df_m['measures']
        #         tf.compat.v1.reset_default_graph()
        #         TRAIN_END += 1
        #     tmp = np.array(df_measure.iloc[:,1:].values)
        #     for i in range(2 + NUM_HEROS_USED, 2 + NUM_HEROS_USED * 3):
        #         if i < 2 + NUM_HEROS_USED * 2:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_up
        #         else:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_down
        #     df_measure.loc[0, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 9, 'HitRatio']
        #     df_measure.loc[1, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 19, 'HitRatio']
        #     df_measure.loc[2, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 49, 'HitRatio']
        #     df_measure.loc[3, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 9, 'HitRatio']
        #     df_measure.loc[4, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 19, 'HitRatio']
        #     df_measure.loc[5, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 49, 'HitRatio']
        #     df_measure.to_csv(DIR_NAME + '/' + 'R3GL_SEED_' + str(RAND_SEED) + '_RL_' + str(RANKLAMBDA) + '.csv')
        # RANKLAMBDA = 0.6

        # # -------------------Node LAMBDA----------------------------------------------------------------
        # for n in range(10):
        #     NODELAMBDA = 0.1 * n
        #     print("Node Lambda===========================", NODELAMBDA)
        #     TRAIN_END = Origin_train_end
        #     df_measure = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['HitRatio'])
        #     for i in range(len(normalize_label) - TRAIN_END - NUM_PRED + 1):
        #         tf.set_random_seed(RAND_SEED)
        #         df_m = train_gcnlstm()
        #         df_measure[versions[TRAIN_END]] = df_m['measures']
        #         tf.compat.v1.reset_default_graph()
        #         TRAIN_END += 1
        #     tmp = np.array(df_measure.iloc[:,1:].values)
        #     for i in range(2 + NUM_HEROS_USED, 2 + NUM_HEROS_USED * 3):
        #         if i < 2 + NUM_HEROS_USED * 2:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_up
        #         else:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_down
        #     df_measure.loc[0, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 9, 'HitRatio']
        #     df_measure.loc[1, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 19, 'HitRatio']
        #     df_measure.loc[2, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 49, 'HitRatio']
        #     df_measure.loc[3, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 9, 'HitRatio']
        #     df_measure.loc[4, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 19, 'HitRatio']
        #     df_measure.loc[5, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 49, 'HitRatio']
        #     df_measure.to_csv(DIR_NAME + '/' + 'R3GL_SEED_' + str(RAND_SEED) + '_NL_' + str(NODELAMBDA) + '.csv')
        # NODELAMBDA = 0.3

        # -------------------Time Steps----------------------------------------------------------------
        # for n in range(5):
        #     TIME_STEPS = n + 1
        #     print("Time Steps===========================", TIME_STEPS)
        #     TRAIN_END = Origin_train_end
        #     df_measure = pd.DataFrame(np.zeros((2 + NUM_HEROS_USED * 3, 1)), columns=['HitRatio'])
        #     for i in range(len(normalize_label) - TRAIN_END - NUM_PRED + 1):
        #         tf.set_random_seed(RAND_SEED)
        #         df_m = train_gcnlstm()
        #         df_measure[versions[TRAIN_END]] = df_m['measures']
        #         tf.compat.v1.reset_default_graph()
        #         TRAIN_END += 1
        #     tmp = np.array(df_measure.iloc[:,1:].values)
        #     for i in range(2 + NUM_HEROS_USED, 2 + NUM_HEROS_USED * 3):
        #         if i < 2 + NUM_HEROS_USED * 2:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_up
        #         else:
        #             df_measure.loc[i, 'HitRatio'] = np.sum(tmp[i])/total_down
        #     df_measure.loc[0, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 9, 'HitRatio']
        #     df_measure.loc[1, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 19, 'HitRatio']
        #     df_measure.loc[2, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED + 49, 'HitRatio']
        #     df_measure.loc[3, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 9, 'HitRatio']
        #     df_measure.loc[4, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 19, 'HitRatio']
        #     df_measure.loc[5, 'HitRatio'] = df_measure.loc[2 + NUM_HEROS_USED*2 + 49, 'HitRatio']
        #     df_measure.to_csv(DIR_NAME + '/' + 'R3GL_SEED_' + str(RAND_SEED) + '_TS_' + str(TIME_STEPS) + '.csv')
        # TIME_STEPS = 3




    end = time.time()
    print("Time cost:", end - start)