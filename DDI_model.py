from tensorflow.compat.v1.keras.layers import *
from tensorflow.compat.v1.keras.regularizers import *
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.optimizers import Adam

import math

import tensorflow.compat.v1.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.compat.v1.keras.backend as K
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from tensorflow.compat.v1.keras.utils import to_categorical


# find feature in the generator
def find_exp(drug_df, ts_exp, column_name):
    return pd.merge(drug_df, ts_exp, left_on=column_name, right_on='pubchem', how='left').iloc[:, 2:]


# Generator
class custom_dataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_label, batch_size, exp_df, shuffle=True):
        self.x = x_set
        self.y = y_label
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.x))
        self.shuffle = shuffle
        self.exp_df = exp_df
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __data_generation__(self, x_list):
        x1 = find_exp(x_list[['drug1']], self.exp_df, 'drug1')
        x2 = find_exp(x_list[['drug2']], self.exp_df, 'drug2')
        x_se = x_list['SE']

        x_se_one_hot = to_categorical(x_list['SE'], num_classes=963)

        x1 = np.array(x1).astype(float)
        x2 = np.array(x2).astype(float)

        return x1, x2, x_se, x_se_one_hot

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x.iloc[indexes]
        batch_y = self.y[indexes]

        x1, x2, x_se, x_se_one_hot = self.__data_generation__(batch_x)

        return [x1, x2, x_se, x_se_one_hot], batch_y


# =============================================================================================
# Model settings
# =============================================================================================

# Checkpoint
class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self, save_path, model_name, init_learining_rate, decay_rate, decay_steps, \
                 save_best_metric='val_loss', this_max=False, **kargs):
        super(CustomModelCheckPoint, self).__init__(**kargs)
        self.epoch_loss = {}
        self.epoch_val_loss = {}
        self.save_path = save_path
        self.model_name = model_name

        self.init_learining_rate = init_learining_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.global_step = 0

        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
        #         print('learning rate: %.5f'%lr)

        metric_value = logs.get(self.save_best_metric)
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_model = self.model
        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_model = self.model

        self.epoch_loss[epoch] = logs.get('loss')
        self.epoch_val_loss[epoch] = logs.get('val_loss')
        self.best_model.save_weights(self.save_path + self.model_name + '.h5')

    def on_epoch_begin(self, epoch, logs={}):
        actual_lr = float(K.get_value(self.model.optimizer.lr))
        decayed_learning_rate = actual_lr * self.decay_rate ** (epoch / self.decay_steps)
        K.set_value(self.model.optimizer.lr, decayed_learning_rate)
        if epoch % 10 == 0:
            K.set_value(self.model.optimizer.lr, self.init_learining_rate)


# =============================================================================================
# Model Evaluation
# =============================================================================================

def mean_predicted_score(true_df, predicted_y, with_plot=True):
    test_pred_result = pd.concat(
        [true_df.reset_index(drop=True), pd.DataFrame(predicted_y, columns=['predicted_score'])], axis=1)

    if (with_plot):
        fig, ax = plt.subplots(figsize=(6, 6))
        temp = test_pred_result.groupby('label')['predicted_score'].apply(list)
        sns.boxplot(x='label', y='predicted_score', data=test_pred_result[['label', 'predicted_score']])
        plt.show()

    return test_pred_result


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def cal_performance(predicted_scores_df):
    uniqueSE = predicted_scores_df.SE.unique()

    dfDict = {elem: pd.DataFrame for elem in uniqueSE}

    for key in dfDict.keys():
        dfDict[key] = predicted_scores_df[:][predicted_scores_df.SE == key]

    se_performance = pd.DataFrame(
        columns=['Side effect no.', 'median_pos', 'median_neg', 'optimal_thr', 'SN', 'SP', 'PR', 'AUC', 'AUPR'])
    for se in uniqueSE:
        df = dfDict[se]

        med_1 = np.median(df[df.label == 1.0].predicted_score)
        med_0 = np.median(df[df.label == 0.0].predicted_score)

        temp_thr = (med_1 + med_0) / 2
        temp_y = df.predicted_score.apply(lambda x: 0 if x > temp_thr else 1)
        tn, fp, fn, tp = confusion_matrix(df.label, temp_y).ravel()

        optimal_thr = Find_Optimal_Cutoff(1 - df.label, df.predicted_score)[0]
        temp_y_opt = df.predicted_score.apply(lambda x: 0 if x > optimal_thr else 1)
        tn, fp, fn, tp = confusion_matrix(df.label, temp_y_opt).ravel()

        auc = roc_auc_score(1 - df.label, df.predicted_score)
        aupr = average_precision_score(1 - df.label, df.predicted_score)

        temp_df = pd.DataFrame(
            {'Side effect no.': se, 'median_pos': med_1, 'median_neg': med_0, 'optimal_thr': optimal_thr, \
             'SN': tp / (tp + fn), 'SP': tn / (tn + fp), 'PR': tp / (tp + fp), 'AUC': auc, 'AUPR': aupr}, index=[0])
        se_performance = pd.concat([se_performance, temp_df], axis=0)

    return se_performance


def calculate_test_performance(predicted_scores_df):
    uniqueSE = predicted_scores_df.SE.unique()

    dfDict = {elem: pd.DataFrame for elem in uniqueSE}

    for key in dfDict.keys():
        dfDict[key] = predicted_scores_df[:][predicted_scores_df.SE == key]

    se_performance = pd.DataFrame(columns=['Side effect no.', 'SN', 'SP', 'PR', 'AUC', 'AUPR'])
    for se in uniqueSE:
        df = dfDict[se]

        tn, fp, fn, tp = confusion_matrix(df.label, df.predicted_label).ravel()

        auc = roc_auc_score(1 - df.label, df.predicted_score)
        aupr = average_precision_score(1 - df.label, df.predicted_score)

        temp_df = pd.DataFrame({'Side effect no.': se, \
                                'SN': tp / (tp + fn), 'SP': tn / (tn + fp), 'PR': tp / (tp + fp), 'AUC': auc,
                                'AUPR': aupr}, index=[0])
        se_performance = pd.concat([se_performance, temp_df], axis=0)

    return se_performance


def calculate_predicted_label_ver3(predicted_score_df, optimal_thr, se_col_name='SE', threshold_col_name='optimal_thr'):
    # 1) 마지막 5개 값 평균
    temp_thr = pd.DataFrame(optimal_thr.iloc[:, -7:-2].mean(axis=1), columns=[threshold_col_name])

    thr = pd.concat([optimal_thr['SE'], temp_thr], axis=1)

    merged = pd.merge(predicted_score_df, thr, left_on='SE', right_on=se_col_name, how='left')
    merged['predicted_label'] = merged['predicted_score'] < merged[threshold_col_name]
    merged.predicted_label = merged.predicted_label.map(int)
    merged['gap'] = merged['predicted_score'] - merged[threshold_col_name]
    merged.gap = merged.gap.map(abs)
    test_perf = merged[['drug1', 'drug2', 'SE', 'label', 'predicted_label', 'predicted_score', 'gap']]
    return test_perf, thr


def external_validation_v2(model, test_x, test_y, exp_df, optimal_threshold, batch_size):
    test_gen = custom_dataGenerator(test_x, test_y.values, batch_size=batch_size, exp_df=exp_df, shuffle=False)
    pred_test = model.predict_generator(generator=test_gen)

    test_prediction_scores = mean_predicted_score(pd.concat([test_x, test_y], axis=1), pred_test)
    test_prediction_predicted_label_df, thr = calculate_predicted_label_ver3(test_prediction_scores, optimal_threshold)
    test_prediction_perf_df = calculate_test_performance(test_prediction_predicted_label_df)

    return test_prediction_predicted_label_df, test_prediction_perf_df, thr


# Calculate average predicted scores & relabel
def merge_both_pairs(ori_predicted_label_df, swi_predicted_label_df, optimal_threshold, thr_col_name):
    merge_label = pd.merge(ori_predicted_label_df, swi_predicted_label_df, left_on=['drug1', 'drug2', 'SE'],
                           right_on=['drug2', 'drug1', 'SE'])[
        ['drug1_x', 'drug2_x', 'SE', 'label_x', 'predicted_label_x', 'predicted_label_y', 'predicted_score_x',
         'predicted_score_y']]
    merge_label['mean_predicted_score'] = (merge_label.predicted_score_x + merge_label.predicted_score_y) / 2
    merge_label.rename(columns={'drug1_x': 'drug1', 'drug2_x': 'drug2', 'SE_x': 'SE', 'label_x': 'label'}, inplace=True)

    merged = pd.merge(merge_label, optimal_threshold, left_on='SE', right_on='SE', how='left')
    merged['final_predicted_label'] = merged['mean_predicted_score'] < merged[thr_col_name]
    merged.final_predicted_label = merged.final_predicted_label.map(int)
    merged['gap'] = merged['mean_predicted_score'] - merged[thr_col_name]
    merged.gap = merged.gap.map(abs)

    merged = merged[['drug1', 'drug2', 'SE', 'label', 'predicted_label_x', 'predicted_label_y', 'predicted_score_x',
                     'predicted_score_y', \
                     'mean_predicted_score', 'final_predicted_label', 'gap']]

    # ======================================================================================
    uniqueSE = merged.SE.unique()

    dfDict = {elem: pd.DataFrame for elem in uniqueSE}

    for key in dfDict.keys():
        dfDict[key] = merged[:][merged.SE == key]

    se_performance = pd.DataFrame(columns=['Side effect no.', 'SN', 'SP', 'PR', 'AUC', 'AUPR'])
    for se in uniqueSE:
        df = dfDict[se]

        tn, fp, fn, tp = confusion_matrix(df.label, df.final_predicted_label).ravel()

        auc = roc_auc_score(1 - df.label, df.mean_predicted_score)
        aupr = average_precision_score(1 - df.label, df.mean_predicted_score)

        temp_df = pd.DataFrame({'Side effect no.': se, \
                                'SN': tp / (tp + fn), 'SP': tn / (tn + fp), 'PR': tp / (tp + fp), 'AUC': auc,
                                'AUPR': aupr}, index=[0])
        se_performance = pd.concat([se_performance, temp_df], axis=0)

    return merged, se_performance

class DDI_model(object):
    def __init__(self, input_drug_dim=978, input_se_dim=1, drug_emb_dim=100, se_emb_dim=100, output_dim=1, margin=1,
                 drug_activation='elu'):
        self.input_drug_dim = input_drug_dim
        self.input_se_dim = input_se_dim
        self.drug_emb_dim = drug_emb_dim
        self.se_emb_dim = se_emb_dim
        self.output_dim = output_dim
        self.margin = margin
        self.drug_activation = drug_activation

        self.callbacks = []
        self.model = self.build()

    def build(self):
        drug1_exp = Input(shape=(self.input_drug_dim,))
        drug2_exp = Input(shape=(self.input_drug_dim,))

        shared_layer = Sequential(name='drug_embed_shared')
        shared_layer.add(Dense(self.input_drug_dim, activation=self.drug_activation))
        shared_layer.add(BatchNormalization())

        drug1 = shared_layer(drug1_exp)
        drug2 = shared_layer(drug2_exp)

        concat = Concatenate()([drug1, drug2])

        glu1 = Dense(self.input_drug_dim, activation='sigmoid', name='drug1_glu')(concat)
        glu2 = Dense(self.input_drug_dim, activation='sigmoid', name='drug2_glu')(concat)

        drug1_selected = Multiply()([drug1, glu1])
        drug2_selected = Multiply()([drug2, glu2])
        drug1_selected = BatchNormalization()(drug1_selected)
        drug2_selected = BatchNormalization()(drug2_selected)

        shared_layer2 = Sequential(name='drug_embed_shared2')
        shared_layer2.add(Dense(self.drug_emb_dim, kernel_regularizer=l2(0.001), activation=self.drug_activation))
        shared_layer2.add(BatchNormalization())

        drug1_emb = shared_layer2(drug1_selected)
        drug2_emb = shared_layer2(drug2_selected)

        # side effect
        input_se = Input(shape=(self.input_se_dim,))
        se_emb = Embedding(963, output_dim=self.se_emb_dim, input_length=self.input_se_dim)(input_se)

        # one-hot side effect for metric
        input_se_one_hot = Input(shape=(963,))

        # side effect mapping matrix
        se_head = Embedding(963, output_dim=self.drug_emb_dim * self.se_emb_dim, input_length=self.input_se_dim,
                            embeddings_regularizer=l2(0.01))(input_se)
        se_head = Reshape((self.se_emb_dim, self.drug_emb_dim))(se_head)
        se_tail = Embedding(963, output_dim=self.drug_emb_dim * self.se_emb_dim, input_length=self.input_se_dim,
                            embeddings_regularizer=l2(0.01))(input_se)
        se_tail = Reshape((self.se_emb_dim, self.drug_emb_dim))(se_tail)

        # MhH & MtT
        mh_dx = Dot(axes=(2, 1))([se_head, drug1_emb])
        mt_dy = Dot(axes=(2, 1))([se_tail, drug2_emb])
        mh_dy = Dot(axes=(2, 1))([se_head, drug2_emb])
        mt_dx = Dot(axes=(2, 1))([se_tail, drug1_emb])

        # || MhH + r - MtT ||
        score1 = add([mh_dx, se_emb])
        score1 = subtract([score1, mt_dy])
        score1 = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1)))(score1)
        score1 = Reshape((1,))(score1)

        score2 = add([mh_dy, se_emb])
        score2 = subtract([score2, mt_dx])
        score2 = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1)))(score2)
        score2 = Reshape((1,))(score2)

        final_score = add([score1, score2])

        model = Model(inputs=[drug1_exp, drug2_exp, input_se, input_se_one_hot], outputs=final_score)
        model.compile(loss=self.custom_loss_wrapper(se_one_hot=input_se_one_hot, margin=self.margin), \
                      optimizer=Adam(0.001), metrics=['accuracy'])

        return model

    def custom_loss_wrapper(self, se_one_hot, margin):
        def custom_margin_loss(y_true, y_pred, se_one_hot=se_one_hot, margin=margin):
            pos_score = (y_true * y_pred)
            neg_score = (K.abs(K.ones_like(y_true) - y_true) * y_pred)

            se_pos = K.dot(K.transpose(pos_score), se_one_hot)
            se_neg = K.dot(K.transpose(neg_score), se_one_hot)

            se_mask = K.cast(se_pos * se_neg, dtype=bool)

            se_pos_score = K.cast(se_mask, dtype='float32') * se_pos
            se_neg_score = K.cast(se_mask, dtype='float32') * se_neg

            score = se_pos_score - se_neg_score + (
                        K.ones_like(se_pos_score) * K.cast(se_mask, dtype='float32')) * margin
            final_loss = K.sum(K.maximum(K.zeros_like(score), score))

            return final_loss

        return custom_margin_loss

    def get_model_summary(self):
        return self.model.summary()

    def set_checkpoint(self):
        checkpoint = CustomModelCheckPoint(save_path=self.model_save_path, model_name=self.model_name, \
                                           init_learining_rate=self.init_lr, decay_rate=self.decay_rate,
                                           decay_steps=self.decay_steps)
        self.callbacks.append(checkpoint)

    def train(self, train_data, exp_df, split_frac, sampling_size, model_save_path, model_name, init_lr=0.0001,
              decay_rate=0.9, decay_steps=2, batch_size=1024):
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.batch_size = batch_size

        self.callbacks = []
        self.set_checkpoint()

        optimal_threshold = pd.DataFrame(np.array(range(0, len(train_data.SE.unique()))), columns=['SE'])

        for n in range(sampling_size):
            print(n, ' Sample =======')
            cv_test = train_data.groupby(['SE', 'label']).apply(pd.DataFrame.sample, frac=split_frac)
            cv_test_x = cv_test.reset_index(drop=True).iloc[:, :3]
            cv_test_y = cv_test.reset_index(drop=True).iloc[:, -1]

            cv_train_data_rest = pd.concat([train_data, cv_test]).drop_duplicates(keep=False, inplace=False)
            cv_train_x = cv_train_data_rest.iloc[:, :3]
            cv_train_y = cv_train_data_rest.iloc[:, 3]
            print('Cross validation train, test dataset size: ', cv_train_x.shape, cv_test_x.shape)

            cv_train_gen = custom_dataGenerator(cv_train_x, cv_train_y.values, batch_size=self.batch_size,
                                                exp_df=exp_df)
            cv_test_gen = custom_dataGenerator(cv_test_x, cv_test_y.values, batch_size=self.batch_size, exp_df=exp_df,
                                               shuffle=False)

            steps_per_epoch = cv_train_x.shape[0] // self.batch_size // 10

            # ======================================================================================================================#
            self.model.fit_generator(generator=cv_train_gen, steps_per_epoch=steps_per_epoch,
                                     validation_data=cv_test_gen, \
                                     epochs=10, verbose=0, shuffle=True, callbacks=self.callbacks)

            cv_test_pred_y = self.model.predict_generator(generator=cv_test_gen)

            cv_test_prediction_scores = mean_predicted_score(cv_test, cv_test_pred_y, with_plot=False)
            cv_test_prediction_perf = cal_performance(cv_test_prediction_scores)
            print('AUC: {:.3f}, AUPR: {:.3f}'.format(cv_test_prediction_perf.describe().loc['mean']['AUC'],
                                                     cv_test_prediction_perf.describe().loc['mean']['AUPR']))

            optimal_threshold = pd.concat(
                [optimal_threshold, pd.DataFrame(cv_test_prediction_perf.optimal_thr).reset_index(drop=True)], axis=1)

        self.optimal_threshold = optimal_threshold
        self.history = self.model.history

    def test(self, test_x, test_y, exp_df):
        switch_x = test_x[['drug2', 'drug1', 'SE']]
        switch_x.columns = ['drug1', 'drug2', 'SE']

        ori_test_prediction_predicted_label_df, ori_test_prediction_perf_df, thr = external_validation_v2(self.model,
                                                                                                          test_x,
                                                                                                          test_y,
                                                                                                          exp_df=exp_df,
                                                                                                          optimal_threshold=self.optimal_threshold,
                                                                                                          batch_size=self.batch_size)
        swi_test_prediction_predicted_label_df, swi_test_prediction_perf_df, thr = external_validation_v2(self.model,
                                                                                                          switch_x,
                                                                                                          test_y,
                                                                                                          exp_df=exp_df,
                                                                                                          optimal_threshold=self.optimal_threshold,
                                                                                                          batch_size=self.batch_size)
        print('Test set predicted === ')

        merge_predicted_label_df, merged_perf_df = merge_both_pairs(ori_test_prediction_predicted_label_df,
                                                                    swi_test_prediction_predicted_label_df, thr,
                                                                    'optimal_thr')
        print('AUC: {:.3f}, AUPR: {:.3f}'.format(merged_perf_df.describe().loc['mean']['AUC'],
                                                 merged_perf_df.describe().loc['mean']['AUPR']))
        return merge_predicted_label_df, merged_perf_df

    def save_model(self):
        self.model.save(self.model_save_path + 'final_{}.h5'.format(self.model_name))
        print('Model saved === ')

    def load_model(self, model_load_path, model_name, threshold_name):
        self.model.load_weights(model_load_path + model_name)
        self.optimal_threshold = pd.read_csv(model_load_path + threshold_name, index_col=0)

    def predict(self, x, exp_df, batch_size=1024):
        y = np.zeros(x.shape[0])

        test_gen = custom_dataGenerator(x, y, batch_size=batch_size, exp_df=exp_df, shuffle=False)
        pred_y = self.model.predict_generator(generator=test_gen)
        predicted_result = mean_predicted_score(pd.concat([x, pd.DataFrame(y, columns=['label'])], axis=1), pred_y,
                                                with_plot=False)
        predicted_label, thr = calculate_predicted_label_ver3(predicted_result, self.optimal_threshold)
        predicted_label = predicted_label[['drug1', 'drug2', 'SE', 'predicted_label', 'predicted_score']]

        return predicted_label
