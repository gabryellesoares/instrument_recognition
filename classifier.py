import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import vggish.vggish as vggish
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

df_openmic = pd.read_csv('openmic/openmic-2018-aggregated-labels.csv')
instruments = np.unique(df_openmic['instrument'])
class_map = {inst: i for i, inst in enumerate(instruments)}

OPENMIC = np.load('openmic/openmic.npz', allow_pickle=True)
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

split_train = pd.read_csv('openmic/partitions/split01_train.csv', header=None).squeeze()
split_test = pd.read_csv('openmic/partitions/split01_test.csv', header=None).squeeze()

print(f'Train: {len(split_train)}\nTest: {len(split_test)}')

train_set = set(split_train)
test_set = set(split_test)

idx_train, idx_test = [], []

for idx, sk in enumerate(sample_key):
    if sk in train_set:
        idx_train.append(idx)
    elif sk in test_set:
        idx_test.append(idx)
        
idx_train = np.asarray(idx_train)
idx_test = np.asarray(idx_test)

X_train = X[idx_train]
X_test = X[idx_test]

Y_true_train = Y_true[idx_train]
Y_true_test = Y_true[idx_test]

Y_mask_train = Y_mask[idx_train]
Y_mask_test = Y_mask[idx_test]

for instrument in class_map:
    inst_num = class_map[instrument]

    train_inst = Y_mask_train[:, inst_num]
    test_inst = Y_mask_test[:, inst_num]
    
    X_train_inst = X_train[train_inst]
    X_train_inst_sklearn = np.mean(X_train_inst, axis=1)
    Y_true_train_inst = Y_true_train[train_inst, inst_num] >= 0.8
    
    X_test_inst = X_test[test_inst]
    X_test_inst_sklearn = np.mean(X_test_inst, axis=1)
    Y_true_test_inst = Y_true_test[test_inst, inst_num] >= 0.8

    clf = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=0)
    
    clf.fit(X_train_inst_sklearn, Y_true_train_inst)

    Y_pred_train = clf.predict(X_train_inst_sklearn)
    Y_pred_test = clf.predict(X_test_inst_sklearn)
    
    print('-' * 52)
    print(instrument)
    print('\tTRAIN')
    print(classification_report(Y_true_train_inst, Y_pred_train))
    print('\tTEST')
    print(classification_report(Y_true_test_inst, Y_pred_test))
        
    with open(f'openmic/models/{instrument}.pkl', 'wb') as file:
        pickle.dump(clf, file)