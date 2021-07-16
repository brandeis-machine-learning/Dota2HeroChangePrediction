import tensorflow as tf
import numpy as np


def gcn_layer(adj, X, W, B):
    cur = tf.matmul(adj, X) #(batch_size, input_size)
    cur = tf.matmul(cur, W)
    cur = tf.nn.bias_add(cur, B)
    return tf.nn.relu(cur)

# calc laplacian matrix, A:N*N
def calculate_laplacian(A):
    adj = A + tf.eye(A.shape.as_list()[0])
    rowsum = tf.reduce_sum(adj, 1)
    d_inv_sqrt = tf.layers.flatten(tf.pow(rowsum, -0.5))
    d_inv_sqrt = tf.reshape(d_inv_sqrt, [-1])
    d_mat_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
    # L = tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    L = tf.matmul(tf.transpose(tf.matmul(adj, d_mat_inv_sqrt)), d_mat_inv_sqrt)
    return L

#-----------------LSTM------------------------------------------------------------
class LSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, fb, cost_matrix):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.rr = rr
        self.fb = fb
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.xs = tf.compat.v1.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        # with tf.compat.v1.variable_scope('in_hidden'):
        #     self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    # def add_input_layer(self,):
    #     out_GCN = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
    #     Ws_in = self._weight_variable([self.input_size, self.cell_size])    # Ws (in_size, cell_size)
    #     bs_in = self._bias_variable([self.cell_size,])  # bs (cell_size, )
    #     with tf.name_scope('Wx_plus_b'):
    #         embedding_LSTM = tf.nn.relu(tf.matmul(out_GCN, Ws_in) + bs_in)   # embedding_LSTM = (batch * n_steps, cell_size)
    #     self.embedding_LSTM = tf.reshape(embedding_LSTM, [-1, self.n_steps, self.cell_size], name='2_3D')   # reshape embedding_LSTM ==> (batch, n_steps, cell_size)

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.xs, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')  # shape = (batch * steps, cell_size)
        Ws_out = Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('Wx_plus_b'):
            predict = tf.nn.relu(tf.matmul(history_LSTM, Ws_out) + bs_out) # shape = (batch * steps, output_size)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')

    def compute_cost(self):
        cross_entropy = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])
        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=lable[t], logits=logit[t]))
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t])))
            self.cost = cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.trainable_variables())










# -----------------------RGCN------------------------------------------------


class RGCN(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)


    # def add_input_layer(self):
    #     W_1 = tf.compat.v1.get_variable(shape=[self.input_size+1, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
    #     B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
    #     W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
    #     B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
    #     with tf.name_scope('GCN_layer'):
    #         PR = tf.nn.softplus(tf.nn.bias_add(tf.matmul(self.adj, W_2), B_2))
    #         norm_RankingPreserve = PR/tf.reduce_sum(PR)
    #         L = calculate_laplacian(self.adj)
    #         out_GCN = gcn_layer(L, tf.concat([norm_RankingPreserve, self.xs],1), W_1, B_1)    # (batch_size, hidden_size)
    #     self.norm_PR = norm_RankingPreserve
    #     self.PR = PR
    #     self.out_GCN = out_GCN

    def add_input_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            PR = tf.nn.softplus(tf.nn.bias_add(tf.matmul(self.adj, W_2), B_2))
            norm_RankingPreserve = PR/tf.reduce_sum(PR)
            A = tf.multiply(self.adj, PR)
            L = calculate_laplacian(A)
            out_GCN = gcn_layer(L, self.xs, W_1, B_1)    # (batch_size, hidden_size)
        self.norm_PR = norm_RankingPreserve
        self.PR = PR
        self.out_GCN = out_GCN

    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            self.pred = tf.nn.relu(tf.matmul(self.out_GCN, Ws_out) + bs_out) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            outI = tf.reduce_sum(self.adj, 1)
            inJ = tf.reduce_sum(self.adj, 0)

            PRMitrixI = tf.multiply(self.norm_PR/outI, tf.ones([1, self.batch_size]))
            PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR/inJ, tf.ones([1, self.batch_size])))
            
            rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

            rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss



# -----------------------GCN------------------------------------------------

class GCN(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        with tf.name_scope('GCN_layer'):
            L = calculate_laplacian(self.adj)
            out_GCN = gcn_layer(L, self.xs, W_1, B_1)    # (batch_size, hidden_size)
        self.out_GCN = out_GCN

    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            self.pred = tf.nn.relu(tf.matmul(self.out_GCN, Ws_out) + bs_out) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0

        with tf.name_scope('average_cost'):
            cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            # cross_entropy += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=self.ys, logits=self.pred, pos_weight=2))
            # cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            self.cost = cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())






# -----------------------GCNLSTM------------------------------------------


class GCNLSTM_base(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)


    def add_input_layer(self):
        # ------------------ Embedding X->GCN ----------------------
        # W_em1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_em1')
        # B_em1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_em1')
        # W_em2 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_em2')
        # B_em2 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_em2')

        # in_GCN = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (steps*batch_size, hidden_size)
        # with tf.name_scope('embedding1'):
        #     embedding1 = tf.nn.relu(tf.matmul(in_GCN, W_em1) + B_em1)   # embedding_LSTM = (steps*batch_size, hidden_size)
        # with tf.name_scope('embedding2'):
        #     embedding2 = tf.nn.relu(tf.matmul(embedding1, W_em2) + B_em2)   # embedding_LSTM = (steps*batch_size, hidden_size)
        # embeded_X = tf.reshape(embedding2, [-1, self.batch_size, self.hidden_size], name='2_3D')   # reshape embedding_LSTM ==> (n_steps, batch_size, hidden_size)


        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for i in range(self.n_steps):
                cur_adj = self.adj[i]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                A = tf.multiply(cur_adj, RankingPreserve)
                L = calculate_laplacian(A)
                H_1 = gcn_layer(L, self.xs[i], W_1, B_1)    # (batch_size, hidden_size)
                H_1 = tf.expand_dims(H_1, 0)
                RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                if i == 0 :
                    out_GCN = H_1
                    PR = RankingPreserve
                else:
                    out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                    PR = tf.concat([PR, RankingPreserve], axis = 0)
        self.PR = PR
        # out_GCN = tf.transpose(tf.concat([out_GCN, self.xs], 2),perm=[1,0,2])  # (batch_size, steps, hidden_size)
        out_GCN = tf.transpose(out_GCN,perm=[1,0,2])  # (batch_size, steps, hidden_size)
        self.embedding_LSTM = out_GCN

        # ------------------ Embedding GCN->LSTM ----------------------
        # out_GCN = tf.reshape(out_GCN, [-1, self.hidden_size], name='2_2D')  # (batch_size*steps, hidden_size)
        # W_em1 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_em1')
        # B_em1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_em1')
        # with tf.name_scope('embedding1'):
        #     embedding1 = tf.nn.relu(tf.matmul(out_GCN, W_em1) + B_em1)   # embedding_LSTM = (batch_size*steps, cell_size)

        # W_em2 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.cell_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_em2')
        # B_em2 = tf.compat.v1.get_variable(shape=[self.cell_size, ], initializer=tf.constant_initializer(0.1), name='B_em2')
        # with tf.name_scope('embedding2'):
        #     embedding2 = tf.nn.relu(tf.matmul(embedding1, W_em2) + B_em2)   # embedding_LSTM = (batch_size*steps, cell_size)
        # self.embedding_LSTM = tf.reshape(embedding2, [-1, self.n_steps, self.cell_size], name='2_3D')   # reshape embedding_LSTM ==> (batch_size, n_steps, cell_size)

    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=self.cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')   # (batch * steps, cell_size)
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            predict = tf.nn.relu(tf.matmul(history_LSTM, Ws_out) + bs_out) # (batch * steps, output_size)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])
        one_batch = tf.ones([1, self.batch_size])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t]))

                # findZeros = np.ones([self.batch_size, self.batch_size])
                # for i in range(self.batch_size):
                #     for j in range(self.batch_size):
                #         if self.adj[t][i][j] == 0:
                #             findZeros[i][j] = 0

                outI = tf.reduce_sum(self.adj[t], 1)
                inJ = tf.reduce_sum(self.adj[t], 0)

                PRMitrixI = tf.multiply(self.PR[t]/outI, one_batch)
                PRMitrixJ = tf.transpose(tf.multiply(self.PR[t]/inJ, one_batch))

                # rankNoEdge = self.nodeLambda * (tf.multiply(self.PR[t], one_batch) + tf.transpose(tf.multiply(self.PR[t], one_batch)))
                # rank = rankEdge * findZeros + rankNoEdge * (1 - findZeros)
                
                rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)
                # rankEdge = inJ * tf.square(PRMitrixJ - PRMitrixI)

                rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t]) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss









# -----------------------RGCNLSTM-------------------------------------------


class RGCNLSTM_base(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for i in range(self.n_steps):
                cur_adj = self.adj[i]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                A = tf.multiply(cur_adj, RankingPreserve)
                L = calculate_laplacian(A)
                H_1 = gcn_layer(L, self.xs[i], W_1, B_1)    # (batch_size, hidden_size)
                H_1 = tf.expand_dims(H_1, 0)
                RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                if i == 0 :
                    out_GCN = H_1
                    PR = RankingPreserve
                else:
                    out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                    PR = tf.concat([PR, RankingPreserve], axis = 0)
        self.PR = PR
        out_GCN = tf.transpose(out_GCN,perm=[1,0,2])  # (batch_size, steps, hidden_size)
        self.embedding_LSTM = out_GCN

    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=self.cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')   # (batch * steps, cell_size)
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            predict = tf.nn.relu(tf.matmul(history_LSTM, Ws_out) + bs_out) # (batch * steps, output_size)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])
        # embed = tf.transpose(self.embedding_LSTM,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t]))
                # cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t])))

                outI = tf.reduce_sum(self.adj[t], 1)
                inJ = tf.reduce_sum(self.adj[t], 0)

                PRMitrixI = tf.multiply(self.PR[t]/outI, tf.ones([1, self.batch_size]))
                PRMitrixJ = tf.transpose(tf.multiply(self.PR[t]/inJ, tf.ones([1, self.batch_size])))

                rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t]) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss










# -----------------------RLSTM-------------------------------------------


class RLSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        # W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        # B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for i in range(self.n_steps):
                cur_adj = self.adj[i]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                # A = tf.multiply(cur_adj, RankingPreserve)
                # L = calculate_laplacian(A)
                # H_1 = gcn_layer(L, self.xs[i], W_1, B_1)    # (batch_size, hidden_size)
                # H_1 = tf.expand_dims(H_1, 0)
                norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)
                norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                if i == 0 :
                    # out_GCN = H_1
                    norm_PR = norm_RankingPreserve
                    PR = RankingPreserve
                else:
                    # out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                    norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)
                    PR = tf.concat([PR, RankingPreserve], axis = 0)
        self.norm_PR = norm_PR
        self.PR = PR
        # out_GCN = tf.transpose(out_GCN,perm=[1,0,2])  # (batch_size, steps, hidden_size)
        out_GCN = tf.transpose(tf.concat([norm_PR, self.xs], 2),perm=[1,0,2])  # (batch_size, steps, hidden_size)
        self.embedding_LSTM = out_GCN

    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=self.cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')   # (batch * steps, cell_size)
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            predict = tf.nn.relu(tf.matmul(history_LSTM, Ws_out) + bs_out) # (batch * steps, output_size)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])
        # embed = tf.transpose(self.embedding_LSTM,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t]))
                # cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t])))

                outI = tf.reduce_sum(self.adj[t], 1)
                inJ = tf.reduce_sum(self.adj[t], 0)

                PRMitrixI = tf.multiply(self.norm_PR[t]/outI, tf.ones([1, self.batch_size]))
                PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[t]/inJ, tf.ones([1, self.batch_size])))

                rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t]) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss






# -----------------------GCNLSTM-------------------------------------------


class GCNLSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        # W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        # B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for i in range(self.n_steps):
                cur_adj = self.adj[i]
                # RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                # A = tf.multiply(cur_adj, RankingPreserve)
                L = calculate_laplacian(cur_adj)
                H_1 = gcn_layer(L, self.xs[i], W_1, B_1)    # (batch_size, hidden_size)
                H_1 = tf.expand_dims(H_1, 0)
                # norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)
                # norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                # RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                if i == 0 :
                    out_GCN = H_1
                    # norm_PR = norm_RankingPreserve
                    # PR = RankingPreserve
                else:
                    out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                    # norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)
                    # PR = tf.concat([PR, RankingPreserve], axis = 0)
        # self.norm_PR = norm_PR
        # self.PR = PR
        out_GCN = tf.transpose(out_GCN, perm=[1,0,2])  # (batch_size, steps, hidden_size)
        self.embedding_LSTM = out_GCN

    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=self.cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')   # (batch * steps, cell_size)
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            predict = tf.nn.relu(tf.matmul(history_LSTM, Ws_out) + bs_out) # (batch * steps, output_size)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        # rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])
        # embed = tf.transpose(self.embedding_LSTM,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t]))
                # cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t])))

                # outI = tf.reduce_sum(self.adj[t], 1)
                # inJ = tf.reduce_sum(self.adj[t], 0)

                # PRMitrixI = tf.multiply(self.norm_PR[t]/outI, tf.ones([1, self.batch_size]))
                # PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[t]/inJ, tf.ones([1, self.batch_size])))

                # rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                # rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t]) - 1)
                

            self.cost = cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            # self.rankLoss = rankLoss






# -----------------------RGCNLSTM-------------------------------------------


class RGCNLSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for i in range(self.n_steps):
                cur_adj = self.adj[i]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                A = tf.multiply(cur_adj, RankingPreserve)
                L = calculate_laplacian(A)
                H_1 = gcn_layer(L, self.xs[i], W_1, B_1)    # (batch_size, hidden_size)
                H_1 = tf.expand_dims(H_1, 0)
                norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)
                norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                if i == 0 :
                    out_GCN = H_1
                    norm_PR = norm_RankingPreserve
                    PR = RankingPreserve
                else:
                    out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                    norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)
                    PR = tf.concat([PR, RankingPreserve], axis = 0)
        self.norm_PR = norm_PR
        self.PR = PR
        out_GCN = tf.transpose(tf.concat([norm_PR, out_GCN], 2),perm=[1,0,2])  # (batch_size, steps, hidden_size)
        self.embedding_LSTM = out_GCN

    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=self.cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')   # (batch * steps, cell_size)
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            predict = tf.nn.relu(tf.matmul(history_LSTM, Ws_out) + bs_out) # (batch * steps, output_size)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])
        # embed = tf.transpose(self.embedding_LSTM,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t]))
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                outI = tf.reduce_sum(self.adj[t], 1)
                inJ = tf.reduce_sum(self.adj[t], 0)

                PRMitrixI = tf.multiply(self.norm_PR[t]/outI, tf.ones([1, self.batch_size]))
                PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[t]/inJ, tf.ones([1, self.batch_size])))

                rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t]) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss







# -----------------------R------------------------------------------------

class R(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            PR = tf.nn.softplus(tf.nn.bias_add(tf.matmul(self.adj, W_2), B_2))
            norm_RankingPreserve = PR/tf.reduce_sum(PR)
            out_R = tf.concat([norm_RankingPreserve, self.xs], axis=1)
        self.norm_PR = norm_RankingPreserve
        self.PR = PR
        self.out_R = out_R

    def add_output_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size+1, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_3 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W3')
        B_3 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B3')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.out_R, W_1) + B_1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_3) + B_3)
            self.pred = tf.nn.relu(tf.matmul(hidden_layer2, Ws_out) + bs_out) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0

        with tf.name_scope('average_cost'):
            cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            # cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            outI = tf.reduce_sum(self.adj, 1)
            inJ = tf.reduce_sum(self.adj, 0)

            PRMitrixI = tf.multiply(self.norm_PR/outI, tf.ones([1, self.batch_size]))
            PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR/inJ, tf.ones([1, self.batch_size])))
            
            rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

            rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss





# -----------------------DNN------------------------------------------------

class DNN(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        # with tf.compat.v1.variable_scope('in_hidden'):
        #     self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)


    def add_output_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B2')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.xs, W_1) + B_1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_2) + B_2)
            self.pred = tf.nn.relu(tf.matmul(hidden_layer2, Ws_out) + bs_out) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            self.cost = cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())









# -----------------------R3GL------------------------------------------------

class R3GL(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                for t in range(self.n_steps):
                    cur_adj = self.adj[t][n]
                    RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                    # A = tf.multiply(cur_adj, RankingPreserve)
                    L = calculate_laplacian(cur_adj)
                    H_1 = gcn_layer(L, self.xs[t], W_1[n], B_1[n])    # (batch_size, hidden_size)
                    norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                    H_1 = tf.expand_dims(H_1, 0)
                    norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                    RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                    if t == 0 :
                        out_GCN = H_1
                        norm_PR = norm_RankingPreserve
                        PR = RankingPreserve
                    else:
                        out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                        norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)  # (steps, batch_size, 1)
                        PR = tf.concat([PR, RankingPreserve], axis = 0)

                if n == 0 :
                    hidden = out_GCN
                    n_PR = norm_PR
                    o_PR = PR
                else:
                    hidden = tf.concat([hidden, out_GCN], axis = 2)   # (steps, batch_size, hidden_size*n_adj)
                    n_PR = tf.concat([n_PR, norm_PR], axis = 2) # (steps, batch_size, n_adj)
                    o_PR = tf.concat([o_PR, PR], axis = 2)

        hidden = tf.concat([n_PR, hidden], axis = 2)  # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.embedding_LSTM = tf.transpose(hidden, perm=[1,0,2])    # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.norm_PR = tf.transpose(n_PR, perm=[0,2,1]) # (steps, n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[0,2,1])


    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=self.cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')   # (batch * steps, cell_size)
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('predict'):
            predict = tf.nn.relu(tf.matmul(history_LSTM, Ws_out) + bs_out) # (batch * steps, output_size)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t]))
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                for n in range(self.n_adj):
                    outI = tf.reduce_sum(self.adj[t][n], 1)
                    inJ = tf.reduce_sum(self.adj[t][n], 0)

                    PRMitrixI = tf.multiply(self.norm_PR[t][n]/outI, tf.ones([1, self.batch_size]))
                    PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[t][n]/inJ, tf.ones([1, self.batch_size])))

                    rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                    rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t][n]) - 1)
                

            self.cost = cross_entropy * self.n_steps + self.rankLambda * rankLoss / self.n_adj + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss




# -----------------------duration_L------------------------------------------------

class Duration_L(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, fb, cost_matrix):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.rr = rr
        self.fb = fb
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.xs = tf.compat.v1.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self,):
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.cell_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.cell_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_3 = tf.compat.v1.get_variable(shape=[self.cell_size, self.cell_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W3')
        B_3 = tf.compat.v1.get_variable(shape=[self.cell_size, ], initializer=tf.constant_initializer(0.1), name='B3')
        W_2 = tf.compat.v1.get_variable(shape=[self.cell_size, self.cell_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[self.cell_size, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('Wx_plus_b'):
            embedding_1 = tf.nn.relu(tf.matmul(self.xs, W_1) + B_1)
            embedding_3 = tf.nn.relu(tf.matmul(embedding_1, W_3) + B_3)
            embedding_2 = tf.nn.relu(tf.matmul(embedding_3, W_2) + B_2)
        self.embd = embedding_2
        
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embd, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        history_LSTM = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_out')
        with tf.name_scope('Wx_plus_b'):
            predict = tf.nn.softplus(tf.matmul(history_LSTM, Ws_out) + bs_out)
            self.pred = tf.reshape(predict, [-1, self.n_steps, self.output_size], name='pred')

    def compute_cost(self):
        cross_entropy = 0
        lable = tf.reshape(self.ys, [self.n_steps, self.output_size])
        logit = tf.reshape(self.pred, [self.n_steps, self.output_size])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                cross_entropy += tf.reduce_mean((lable[t]-logit[t])**2) * np.math.pow(self.fb, self.n_steps - t - 1)
                # cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t])))
            self.cost = cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.trainable_variables())








# # -----------------------R3GL_Rank------------------------------------------------

class R3GL_Rank(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        self.layers_dims = []
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.init_c = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_c')
            self.init_h = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_h')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.n_adj, self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[self.n_adj, 1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                for t in range(self.n_steps):
                    cur_adj = self.adj[t][n]
                    RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2[n]), B_2[n]))
                    # A = tf.multiply(cur_adj, RankingPreserve)
                    L = calculate_laplacian(cur_adj)
                    H_1 = gcn_layer(L, self.xs[t], W_1, B_1)    # (batch_size, hidden_size)
                    norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                    H_1 = tf.expand_dims(H_1, 0)
                    norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                    RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                    if t == 0 :
                        out_GCN = H_1
                        norm_PR = norm_RankingPreserve
                        PR = RankingPreserve
                    else:
                        out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                        norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)  # (steps, batch_size, 1)
                        PR = tf.concat([PR, RankingPreserve], axis = 0)

                if n == 0 :
                    hidden = out_GCN
                    n_PR = norm_PR
                    o_PR = PR
                else:
                    hidden = tf.concat([hidden, out_GCN], axis = 2)   # (steps, batch_size, hidden_size*n_adj)
                    n_PR = tf.concat([n_PR, norm_PR], axis = 2) # (steps, batch_size, n_adj)
                    o_PR = tf.concat([o_PR, PR], axis = 2)

        hidden = tf.concat([n_PR, hidden], axis = 2)  # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.embedding_LSTM = tf.transpose(hidden, perm=[1,0,2])    # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.norm_PR = tf.transpose(n_PR, perm=[0,2,1]) # (steps, n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[0,2,1])


    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            # self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_init_state = tf.contrib.rnn.LSTMStateTuple(c=self.init_c, h=self.init_h)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            self.output_rank = tf.nn.softplus(tf.matmul(self.cell_outputs, Ws_out) + bs_out) # (batch, steps, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + bs_pred)  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t])) * np.math.pow(self.fb, self.n_steps - t - 1)
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                for n in range(self.n_adj):
                    outI = tf.reduce_sum(self.adj[t][n], 1)
                    inJ = tf.reduce_sum(self.adj[t][n], 0)

                    PRMitrixI = tf.multiply(self.PR[t][n]/outI, tf.ones([1, self.batch_size]))
                    PRMitrixJ = tf.transpose(tf.multiply(self.PR[t][n]/inJ, tf.ones([1, self.batch_size])))

                    rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                    rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t][n]) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss / self.n_adj + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss














# -----------------------RL_Rank------------------------------------------------

class RL_Rank(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.init_c = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_c')
            self.init_h = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_h')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        # W_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        # B_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                for t in range(self.n_steps):
                    cur_adj = self.adj[t][n]
                    RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                    norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                    norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                    RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                    if t == 0 :
                        norm_PR = norm_RankingPreserve
                        PR = RankingPreserve
                    else:
                        norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)  # (steps, batch_size, 1)
                        PR = tf.concat([PR, RankingPreserve], axis = 0)

                if n == 0 :
                    n_PR = norm_PR
                    o_PR = PR
                else:
                    n_PR = tf.concat([n_PR, norm_PR], axis = 2) # (steps, batch_size, n_adj)
                    o_PR = tf.concat([o_PR, PR], axis = 2)

        hidden = tf.concat([n_PR, self.xs], axis = 2)  # (steps, batch_size, input_size+n_adj)
        self.embedding_LSTM = hidden
        self.norm_PR = tf.transpose(n_PR, perm=[0,2,1]) # (steps, n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[0,2,1])


    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            # if not hasattr(self, 'cell_init_state'):
            #     self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_init_state = tf.contrib.rnn.LSTMStateTuple(c=self.init_c, h=self.init_h)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, tf.transpose(self.embedding_LSTM, perm=[1,0,2]), initial_state=cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            self.output_rank = tf.nn.softplus(tf.matmul(self.cell_outputs, Ws_out) + bs_out) # (batch, steps, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + bs_pred)  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t])) * np.math.pow(self.fb, self.n_steps - t - 1)
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                # for n in range(self.n_adj):
                #     outI = tf.reduce_sum(self.adj[t][n], 1)
                #     inJ = tf.reduce_sum(self.adj[t][n], 0)

                #     PRMitrixI = tf.multiply(self.norm_PR[t][n]/outI, tf.ones([1, self.batch_size]))
                #     PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[t][n]/inJ, tf.ones([1, self.batch_size])))

                #     rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                #     rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t][n]) - 1)
                

            self.cost = cross_entropy * self.n_steps + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables()) #+ self.rankLambda * rankLoss / self.n_adj
            self.rankLoss = cross_entropy









# -----------------------DNN_rank------------------------------------------------

class DNN_rank(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        # with tf.compat.v1.variable_scope('in_hidden'):
        #     self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)


    def add_output_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B2')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        Bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        Bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.xs, W_1) + B_1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_2) + B_2)
            self.output_rank = tf.nn.softplus(tf.matmul(hidden_layer2, Ws_out) + Bs_out) # (batch, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + Bs_pred) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            self.cost = cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())





# -----------------------R_rank------------------------------------------------

class R_rank(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                cur_adj = self.adj[n]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                if n == 0 :
                    n_PR = norm_RankingPreserve
                    o_PR = RankingPreserve
                else:
                    n_PR = tf.concat([n_PR, norm_RankingPreserve], axis = 1) # (batch_size, n_adj)
                    o_PR = tf.concat([o_PR, RankingPreserve], axis = 1)

        hidden = tf.concat([n_PR, self.xs], axis = 1)  # (batch_size,  input_size+n_adj)
        self.embedding_LSTM = hidden
        self.norm_PR = tf.transpose(n_PR, perm=[1,0]) # (n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[1,0])

    def add_output_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size+self.n_adj, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_3 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W3')
        B_3 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B3')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        Bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        Bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.embedding_LSTM, W_1) + B_1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_3) + B_3)
            self.output_rank = tf.nn.softplus(tf.matmul(hidden_layer2, Ws_out) + Bs_out) # (batch, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + Bs_pred) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            for n in range(self.n_adj):
                outI = tf.reduce_sum(self.adj[n], 1)
                inJ = tf.reduce_sum(self.adj[n], 0)

                PRMitrixI = tf.multiply(self.norm_PR[n]/outI, tf.ones([1, self.batch_size]))
                PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[n]/inJ, tf.ones([1, self.batch_size])))

                rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[n]) - 1)
                

            self.cost = cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss




# -----------------------RG_rank------------------------------------------------

class RG_rank(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                cur_adj = self.adj[n]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)
                L = calculate_laplacian(cur_adj)
                H_1 = gcn_layer(L, self.xs, W_1[n], B_1[n]) # (batch_size, hidden_size)
                if n == 0 :
                    n_PR = norm_RankingPreserve
                    o_PR = RankingPreserve
                    out_GCN = H_1
                else:
                    n_PR = tf.concat([n_PR, norm_RankingPreserve], axis = 1) # (batch_size, n_adj)
                    o_PR = tf.concat([o_PR, RankingPreserve], axis = 1)
                    out_GCN = tf.concat([out_GCN, H_1], axis = 1)   # (batch_size, hidden_size*n_adj)

        hidden = tf.concat([n_PR, out_GCN], axis = 1)  # (batch_size, n_adj+hidden_size*n_adj)
        self.embedding = out_GCN
        self.norm_PR = tf.transpose(n_PR, perm=[1,0]) # (n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[1,0])

    def add_output_layer(self):
        W_H1 = tf.compat.v1.get_variable(shape=[self.hidden_size*self.n_adj, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_H1')
        B_H1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_H1')
        W_H2 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_H2')
        B_H2 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_H2')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        Bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        Bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.embedding, W_H1) + B_H1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_H2) + B_H2)
            self.output_rank = tf.nn.softplus(tf.matmul(hidden_layer2, Ws_out) + Bs_out) # (batch, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + Bs_pred) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            # for n in range(self.n_adj):
            #     outI = tf.reduce_sum(self.adj[n], 1)
            #     inJ = tf.reduce_sum(self.adj[n], 0)

            #     PRMitrixI = tf.multiply(self.norm_PR[n]/outI, tf.ones([1, self.batch_size]))
            #     PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[n]/inJ, tf.ones([1, self.batch_size])))

            #     rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

            #     rankLoss += tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[n]) - 1)
                

            self.cost = cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())# + self.rankLambda * rankLoss
            self.rankLoss = self.cost































# # -----------------------R3GL_Rank------------------------------------------------

class R3GL_Rank_parameter(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        self.layers_dims = []
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.init_c = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_c')
            self.init_h = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_h')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.n_adj, self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[self.n_adj, 1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                for t in range(self.n_steps):
                    cur_adj = self.adj[t][n]
                    RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2[n]), B_2[n]))
                    # A = tf.multiply(cur_adj, RankingPreserve)
                    L = calculate_laplacian(cur_adj)
                    H_1 = gcn_layer(L, self.xs[t], W_1, B_1)    # (batch_size, hidden_size)
                    norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                    H_1 = tf.expand_dims(H_1, 0)
                    norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                    RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                    if t == 0 :
                        out_GCN = H_1
                        norm_PR = norm_RankingPreserve
                        PR = RankingPreserve
                    else:
                        out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                        norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)  # (steps, batch_size, 1)
                        PR = tf.concat([PR, RankingPreserve], axis = 0)

                if n == 0 :
                    hidden = out_GCN
                    n_PR = norm_PR
                    o_PR = PR
                else:
                    hidden = tf.concat([hidden, out_GCN], axis = 2)   # (steps, batch_size, hidden_size*n_adj)
                    n_PR = tf.concat([n_PR, norm_PR], axis = 2) # (steps, batch_size, n_adj)
                    o_PR = tf.concat([o_PR, PR], axis = 2)

        hidden = tf.concat([n_PR, hidden], axis = 2)  # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.embedding_LSTM = tf.transpose(hidden, perm=[1,0,2])    # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.norm_PR = tf.transpose(n_PR, perm=[0,2,1]) # (steps, n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[0,2,1])


    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            # self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_init_state = tf.contrib.rnn.LSTMStateTuple(c=self.init_c, h=self.init_h)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            self.output_rank = tf.nn.softplus(tf.matmul(self.cell_outputs, Ws_out) + bs_out) # (batch, steps, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + bs_pred)  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t])) * np.math.pow(self.fb, self.n_steps - t - 1)
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                for n in range(self.n_adj):
                    outI = tf.reduce_sum(self.adj[t][n], 1)
                    inJ = tf.reduce_sum(self.adj[t][n], 0)

                    PRMitrixI = tf.multiply(self.PR[t][n]/outI, tf.ones([1, self.batch_size]))
                    PRMitrixJ = tf.transpose(tf.multiply(self.PR[t][n]/inJ, tf.ones([1, self.batch_size])))

                    rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                    rankLoss += (1 - self.nodeLambda) * tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t][n]) - 1)
                

            self.cost = (1 - self.rankLambda) * cross_entropy + self.rankLambda * rankLoss / self.n_adj + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss



# -----------------------R_rank------------------------------------------------

class R_rank_parameter(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                cur_adj = self.adj[n]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                if n == 0 :
                    n_PR = norm_RankingPreserve
                    o_PR = RankingPreserve
                else:
                    n_PR = tf.concat([n_PR, norm_RankingPreserve], axis = 1) # (batch_size, n_adj)
                    o_PR = tf.concat([o_PR, RankingPreserve], axis = 1)

        hidden = tf.concat([n_PR, self.xs], axis = 1)  # (batch_size,  input_size+n_adj)
        self.embedding_LSTM = hidden
        self.norm_PR = tf.transpose(n_PR, perm=[1,0]) # (n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[1,0])

    def add_output_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size+self.n_adj, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_3 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W3')
        B_3 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B3')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        Bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        Bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.embedding_LSTM, W_1) + B_1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_3) + B_3)
            self.output_rank = tf.nn.softplus(tf.matmul(hidden_layer2, Ws_out) + Bs_out) # (batch, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + Bs_pred) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            for n in range(self.n_adj):
                outI = tf.reduce_sum(self.adj[n], 1)
                inJ = tf.reduce_sum(self.adj[n], 0)

                PRMitrixI = tf.multiply(self.norm_PR[n]/outI, tf.ones([1, self.batch_size]))
                PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[n]/inJ, tf.ones([1, self.batch_size])))

                rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                rankLoss += (1 - self.nodeLambda) * tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[n]) - 1)
                

            self.cost = (1 - self.rankLambda) * cross_entropy + self.rankLambda * rankLoss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())
            self.rankLoss = rankLoss




# -----------------------RG_rank------------------------------------------------

class RG_rank_parameter(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                cur_adj = self.adj[n]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)
                L = calculate_laplacian(cur_adj)
                H_1 = gcn_layer(L, self.xs, W_1[n], B_1[n]) # (batch_size, hidden_size)
                if n == 0 :
                    n_PR = norm_RankingPreserve
                    o_PR = RankingPreserve
                    out_GCN = H_1
                else:
                    n_PR = tf.concat([n_PR, norm_RankingPreserve], axis = 1) # (batch_size, n_adj)
                    o_PR = tf.concat([o_PR, RankingPreserve], axis = 1)
                    out_GCN = tf.concat([out_GCN, H_1], axis = 1)   # (batch_size, hidden_size*n_adj)

        hidden = tf.concat([n_PR, out_GCN], axis = 1)  # (batch_size, n_adj+hidden_size*n_adj)
        self.embedding = out_GCN
        self.norm_PR = tf.transpose(n_PR, perm=[1,0]) # (n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[1,0])

    def add_output_layer(self):
        W_H1 = tf.compat.v1.get_variable(shape=[self.hidden_size*self.n_adj, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_H1')
        B_H1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_H1')
        W_H2 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_H2')
        B_H2 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_H2')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        Bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        Bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.embedding, W_H1) + B_H1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_H2) + B_H2)
            self.output_rank = tf.nn.softplus(tf.matmul(hidden_layer2, Ws_out) + Bs_out) # (batch, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + Bs_pred) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            for n in range(self.n_adj):
                outI = tf.reduce_sum(self.adj[n], 1)
                inJ = tf.reduce_sum(self.adj[n], 0)

                PRMitrixI = tf.multiply(self.norm_PR[n]/outI, tf.ones([1, self.batch_size]))
                PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[n]/inJ, tf.ones([1, self.batch_size])))

                rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                rankLoss += (1 - self.nodeLambda) * tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[n]) - 1)
                

            self.cost = (1 - self.rankLambda) * cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables()) + self.rankLambda * rankLoss
            self.rankLoss = rankLoss



# -----------------------G_rank------------------------------------------------

class G_rank_parameter(object):
    def __init__(self, input_size, output_size, batch_size, lr, rr, hidden_size, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [batch_size, output_size], name='ys')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        W_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                cur_adj = self.adj[n]
                RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)
                L = calculate_laplacian(cur_adj)
                H_1 = gcn_layer(L, self.xs, W_1[n], B_1[n]) # (batch_size, hidden_size)
                if n == 0 :
                    n_PR = norm_RankingPreserve
                    o_PR = RankingPreserve
                    out_GCN = H_1
                else:
                    n_PR = tf.concat([n_PR, norm_RankingPreserve], axis = 1) # (batch_size, n_adj)
                    o_PR = tf.concat([o_PR, RankingPreserve], axis = 1)
                    out_GCN = tf.concat([out_GCN, H_1], axis = 1)   # (batch_size, hidden_size*n_adj)

        hidden = tf.concat([n_PR, out_GCN], axis = 1)  # (batch_size, n_adj+hidden_size*n_adj)
        self.embedding = out_GCN
        self.norm_PR = tf.transpose(n_PR, perm=[1,0]) # (n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[1,0])

    def add_output_layer(self):
        W_H1 = tf.compat.v1.get_variable(shape=[self.hidden_size*self.n_adj, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_H1')
        B_H1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_H1')
        W_H2 = tf.compat.v1.get_variable(shape=[self.hidden_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_H2')
        B_H2 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B_H2')
        Ws_out = tf.compat.v1.get_variable(shape=[self.hidden_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        Bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        Bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            hidden_layer = tf.nn.relu(tf.matmul(self.embedding, W_H1) + B_H1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, W_H2) + B_H2)
            self.output_rank = tf.nn.softplus(tf.matmul(hidden_layer2, Ws_out) + Bs_out) # (batch, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + Bs_pred) # (batch, output_size)


    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0

        with tf.name_scope('average_cost'):
            # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred))
            cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(self.ys, self.cost_matrix), tf.nn.softmax(self.pred)))

            # for n in range(self.n_adj):
            #     outI = tf.reduce_sum(self.adj[n], 1)
            #     inJ = tf.reduce_sum(self.adj[n], 0)

            #     PRMitrixI = tf.multiply(self.norm_PR[n]/outI, tf.ones([1, self.batch_size]))
            #     PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[n]/inJ, tf.ones([1, self.batch_size])))

            #     rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

            #     rankLoss += (1 - self.nodeLambda) * tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[n]) - 1)
                

            self.cost = (1 - self.rankLambda) * cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())# + self.rankLambda * rankLoss
            self.rankLoss = self.cost






# -----------------------RL_Rank------------------------------------------------

class RL_Rank_parameter(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.init_c = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_c')
            self.init_h = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_h')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        # W_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        # B_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                for t in range(self.n_steps):
                    cur_adj = self.adj[t][n]
                    RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                    norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                    norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                    RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                    if t == 0 :
                        norm_PR = norm_RankingPreserve
                        PR = RankingPreserve
                    else:
                        norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)  # (steps, batch_size, 1)
                        PR = tf.concat([PR, RankingPreserve], axis = 0)

                if n == 0 :
                    n_PR = norm_PR
                    o_PR = PR
                else:
                    n_PR = tf.concat([n_PR, norm_PR], axis = 2) # (steps, batch_size, n_adj)
                    o_PR = tf.concat([o_PR, PR], axis = 2)

        hidden = tf.concat([n_PR, self.xs], axis = 2)  # (steps, batch_size, input_size+n_adj)
        self.embedding_LSTM = hidden
        self.norm_PR = tf.transpose(n_PR, perm=[0,2,1]) # (steps, n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[0,2,1])


    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            # if not hasattr(self, 'cell_init_state'):
            #     self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_init_state = tf.contrib.rnn.LSTMStateTuple(c=self.init_c, h=self.init_h)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, tf.transpose(self.embedding_LSTM, perm=[1,0,2]), initial_state=cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            self.output_rank = tf.nn.softplus(tf.matmul(self.cell_outputs, Ws_out) + bs_out) # (batch, steps, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + bs_pred)  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t])) * np.math.pow(self.fb, self.n_steps - t - 1)
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                for n in range(self.n_adj):
                    outI = tf.reduce_sum(self.adj[t][n], 1)
                    inJ = tf.reduce_sum(self.adj[t][n], 0)

                    PRMitrixI = tf.multiply(self.norm_PR[t][n]/outI, tf.ones([1, self.batch_size]))
                    PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[t][n]/inJ, tf.ones([1, self.batch_size])))

                    rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                    rankLoss += (1 - self.nodeLambda) * tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t][n]) - 1)
                

            self.cost = (1 - self.rankLambda) * cross_entropy * self.n_steps + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables()) + self.rankLambda * rankLoss / self.n_adj
            self.rankLoss = rankLoss




# -----------------------L_Rank------------------------------------------------

class L_Rank_parameter(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.init_c = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_c')
            self.init_h = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_h')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        # W_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        # B_1 = tf.compat.v1.get_variable(shape=[self.n_adj, self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                for t in range(self.n_steps):
                    cur_adj = self.adj[t][n]
                    RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2), B_2))
                    norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                    norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                    RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                    if t == 0 :
                        norm_PR = norm_RankingPreserve
                        PR = RankingPreserve
                    else:
                        norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)  # (steps, batch_size, 1)
                        PR = tf.concat([PR, RankingPreserve], axis = 0)

                if n == 0 :
                    n_PR = norm_PR
                    o_PR = PR
                else:
                    n_PR = tf.concat([n_PR, norm_PR], axis = 2) # (steps, batch_size, n_adj)
                    o_PR = tf.concat([o_PR, PR], axis = 2)

        hidden = tf.concat([n_PR, self.xs], axis = 2)  # (steps, batch_size, input_size+n_adj)
        self.embedding_LSTM = hidden
        self.norm_PR = tf.transpose(n_PR, perm=[0,2,1]) # (steps, n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[0,2,1])


    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            # if not hasattr(self, 'cell_init_state'):
            #     self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_init_state = tf.contrib.rnn.LSTMStateTuple(c=self.init_c, h=self.init_h)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, tf.transpose(self.embedding_LSTM, perm=[1,0,2]), initial_state=cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            self.output_rank = tf.nn.softplus(tf.matmul(self.cell_outputs, Ws_out) + bs_out) # (batch, steps, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + bs_pred)  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t])) * np.math.pow(self.fb, self.n_steps - t - 1)
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                # for n in range(self.n_adj):
                #     outI = tf.reduce_sum(self.adj[t][n], 1)
                #     inJ = tf.reduce_sum(self.adj[t][n], 0)

                #     PRMitrixI = tf.multiply(self.norm_PR[t][n]/outI, tf.ones([1, self.batch_size]))
                #     PRMitrixJ = tf.transpose(tf.multiply(self.norm_PR[t][n]/inJ, tf.ones([1, self.batch_size])))

                #     rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                #     rankLoss += (1 - self.nodeLambda) * tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t][n]) - 1)
                

            self.cost = (1 - self.rankLambda) * cross_entropy * self.n_steps + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables())# + self.rankLambda * rankLoss / self.n_adj
            self.rankLoss = self.cost





# # -----------------------GL_Rank------------------------------------------------

class GL_Rank_parameter(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, rr, hidden_size, forget_bias, nodeLambda, rankLambda, cost_matrix, n_adj):
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.fb = forget_bias
        self.rr = rr
        self.nodeLambda = nodeLambda
        self.rankLambda = rankLambda
        self.cost_matrix = tf.cast(cost_matrix, tf.float32)
        self.n_adj = n_adj
        self.layers_dims = []
        with tf.name_scope('inputs'):
            self.adj = tf.compat.v1.placeholder(tf.float32, [n_steps, n_adj, batch_size, batch_size], name='adj')
            self.xs = tf.compat.v1.placeholder(tf.float32, [n_steps, batch_size, input_size], name='xs')
            self.ys = tf.compat.v1.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.init_c = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_c')
            self.init_h = tf.compat.v1.placeholder(tf.float32, [batch_size, cell_size], name='init_h')
        with tf.compat.v1.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.compat.v1.variable_scope('LSTMGCN_cell'):
            self.add_cell()
        with tf.compat.v1.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def add_input_layer(self):
        # ------------------ Ranking Preservation & GCN ----------------------
        W_1 = tf.compat.v1.get_variable(shape=[self.input_size, self.hidden_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W1')
        B_1 = tf.compat.v1.get_variable(shape=[self.hidden_size, ], initializer=tf.constant_initializer(0.1), name='B1')
        W_2 = tf.compat.v1.get_variable(shape=[self.n_adj, self.batch_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W2')
        B_2 = tf.compat.v1.get_variable(shape=[self.n_adj, 1, ], initializer=tf.constant_initializer(0.1), name='B2')
        with tf.name_scope('GCN_layer'):
            for n in range(self.n_adj):
                for t in range(self.n_steps):
                    cur_adj = self.adj[t][n]
                    RankingPreserve = tf.nn.softplus(tf.nn.bias_add(tf.matmul(cur_adj, W_2[n]), B_2[n]))
                    # A = tf.multiply(cur_adj, RankingPreserve)
                    L = calculate_laplacian(cur_adj)
                    H_1 = gcn_layer(L, self.xs[t], W_1, B_1)    # (batch_size, hidden_size)
                    norm_RankingPreserve = RankingPreserve/tf.reduce_sum(RankingPreserve)

                    H_1 = tf.expand_dims(H_1, 0)
                    norm_RankingPreserve = tf.expand_dims(norm_RankingPreserve, 0)
                    RankingPreserve = tf.expand_dims(RankingPreserve, 0)
                    if t == 0 :
                        out_GCN = H_1
                        norm_PR = norm_RankingPreserve
                        PR = RankingPreserve
                    else:
                        out_GCN = tf.concat([out_GCN, H_1], axis = 0)   # (steps, batch_size, hidden_size)
                        norm_PR = tf.concat([norm_PR, norm_RankingPreserve], axis = 0)  # (steps, batch_size, 1)
                        PR = tf.concat([PR, RankingPreserve], axis = 0)

                if n == 0 :
                    hidden = out_GCN
                    n_PR = norm_PR
                    o_PR = PR
                else:
                    hidden = tf.concat([hidden, out_GCN], axis = 2)   # (steps, batch_size, hidden_size*n_adj)
                    n_PR = tf.concat([n_PR, norm_PR], axis = 2) # (steps, batch_size, n_adj)
                    o_PR = tf.concat([o_PR, PR], axis = 2)

        hidden = tf.concat([n_PR, hidden], axis = 2)  # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.embedding_LSTM = tf.transpose(hidden, perm=[1,0,2])    # (batch_size, steps, hidden_size*n_adj+n_adj)
        self.norm_PR = tf.transpose(n_PR, perm=[0,2,1]) # (steps, n_adj, batch_size)
        self.PR = tf.transpose(o_PR, perm=[0,2,1])


    # ------------------ LSTM ----------------------
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=self.fb, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            # self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_init_state = tf.contrib.rnn.LSTMStateTuple(c=self.init_c, h=self.init_h)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.embedding_LSTM, initial_state=cell_init_state, time_major=False)


    # ------------------ LSTM->output ----------------------
    def add_output_layer(self):
        Ws_out = tf.compat.v1.get_variable(shape=[self.cell_size, 1], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_out')
        bs_out = tf.compat.v1.get_variable(shape=[1, ], initializer=tf.constant_initializer(0.1), name='B_out')
        Ws_pred = tf.compat.v1.get_variable(shape=[1, self.output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.), name='W_pred')
        bs_pred = tf.compat.v1.get_variable(shape=[self.output_size, ], initializer=tf.constant_initializer(0.1), name='B_pred')
        with tf.name_scope('predict'):
            self.output_rank = tf.nn.softplus(tf.matmul(self.cell_outputs, Ws_out) + bs_out) # (batch, steps, 1)
            self.pred = tf.nn.relu(tf.matmul(self.output_rank, Ws_pred) + bs_pred)  # (batch, steps, output_size)

    def compute_cost(self):
        cross_entropy = 0
        rankLoss = 0
        lable = tf.transpose(self.ys,perm=[1,0,2])
        logit = tf.transpose(self.pred,perm=[1,0,2])

        with tf.name_scope('average_cost'):
            for t in range(self.n_steps):
                # cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable[t], logits=logit[t])) * np.math.pow(self.fb, self.n_steps - t - 1)
                cross_entropy += tf.reduce_mean(tf.multiply(tf.matmul(lable[t], self.cost_matrix), tf.nn.softmax(logit[t]))) * np.math.pow(self.fb, self.n_steps - t - 1)

                # for n in range(self.n_adj):
                #     outI = tf.reduce_sum(self.adj[t][n], 1)
                #     inJ = tf.reduce_sum(self.adj[t][n], 0)

                #     PRMitrixI = tf.multiply(self.PR[t][n]/outI, tf.ones([1, self.batch_size]))
                #     PRMitrixJ = tf.transpose(tf.multiply(self.PR[t][n]/inJ, tf.ones([1, self.batch_size])))

                #     rankEdge = -1 * tf.math.log_sigmoid(PRMitrixJ - PRMitrixI)

                #     rankLoss += (1 - self.nodeLambda) * tf.reduce_mean(rankEdge) + self.nodeLambda * (tf.math.reduce_sum(self.PR[t][n]) - 1)
                

            self.cost = (1 - self.rankLambda) * cross_entropy + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.rr), tf.compat.v1.trainable_variables()) #+ self.rankLambda * rankLoss / self.n_adj
            self.rankLoss = self.cost