from matplotlib.backends import registry
import random
import os
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import StratifiedKFold, train_test_split

def split_data(data_file, output_dir, mode):

    X,y = load_svmlight_file(data_file)

    if mode == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        dump_svmlight_file(X_train, y_train, os.path.join(output_dir, 'holdout', 'train.scale'))
        dump_svmlight_file(X_test, y_test, os.path.join(output_dir, 'holdout', 'test.scale'))

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
            dump_svmlight_file(X_train_fold, y_train_fold, os.path.join(output_dir, 'cross_validation', f'train_fold{fold}.scale'))
            dump_svmlight_file(X_val_fold, y_val_fold, os.path.join(output_dir, 'cross_validation', f'val_fold{fold}.scale'))
        


if __name__ == "__main__":
    split_data('pokemon_data.scale', 'data', 'holdout')
    split_data('pokemon_data.scale', 'data', 'cv')