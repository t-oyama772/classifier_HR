import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.logger import get_logger

def main():

    logger = get_logger("hr")
    logger.info('start!')

    # pipeline setting
    pipelines = {
        'knn': Pipeline([('scl',StandardScaler()),
                            ('est',KNeighborsClassifier())]),
            
        'logistic': Pipeline([('scl',StandardScaler()),
                            ('est',LogisticRegression(random_state=1))]),
            
        'rsvc': Pipeline([('scl',StandardScaler()),
                            ('est',SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=1))]),
            
        'lsvc': Pipeline([('scl',StandardScaler()),
                            ('est',LinearSVC(C=1.0, class_weight='balanced', random_state=1))]),
            
        'tree': Pipeline([('scl',StandardScaler()),
                            ('est',DecisionTreeClassifier(random_state=1))]),
            
        'rf': Pipeline([('scl',StandardScaler()),
                            ('est',RandomForestClassifier(random_state=1))]),
            
        'gb': Pipeline([('scl',StandardScaler()),
                            ('est',GradientBoostingClassifier(random_state=1))]),
            
        'mlp': Pipeline([('scl',StandardScaler()),
                            ('est',MLPClassifier(hidden_layer_sizes=(3,3), max_iter=1000, random_state=1))])
    }

    # data load
    my_dtype = {'sales':object, 'salary':object}
    df = pd.read_csv('../data/' + file_model + '.csv', header=0, dtype=my_dtype)

    # sort columns
    col = df.columns.tolist()
    col.remove('left')
    col.append('left')
    df2 = df.loc[:,col]

    ID = df2.iloc[:,0] 
    y = df2.iloc[:,-1]
    X = df2.iloc[:,1:-1]

    # check the shape
    logger.info('---------------------------------')
    logger.info('Raw shape: (%i,%i)' %df2.shape)
    logger.info('X shape: (%i,%i)' %X.shape)
    logger.info('---------------------------------')
    logger.info(X.dtypes)

    logger.info('-----df2------')
    logger.info(df2.head(5))
    logger.info('-----X------')
    logger.info(X.head(5))

    logger.info('-----check null exists------')
    logger.info(df2.isnull().any())

    # preprocessing-1: one-hot encoding （カテゴリ変数の数値化）
    X_ohe = pd.get_dummies(X, dummy_na=True, columns=ohe_cols)
    X_ohe = X_ohe.dropna(axis=1, how='all')
    X_ohe_columns = X_ohe.columns.values

    logger.info('-----------after one-hot encoding----------------------')
    logger.info(X_ohe.head(5))

    from sklearn.impute import SimpleImputer

    # preprocessing-2: null imputation （欠損値対応（平均値で置換される））
    imp = SimpleImputer()
    imp.fit(X_ohe)
    X_ohe = pd.DataFrame(imp.transform(X_ohe), columns=X_ohe_columns)
    logger.info('-----------after null imputation----------------------')
    logger.info(X_ohe.shape)

    logger.info('-----check null not exists------')
    logger.info(X_ohe.isnull().any())


    # preprocessing-3: feature selection （次元圧縮（特徴量削減））
    selector = RFECV(estimator=RandomForestClassifier(random_state=0),step=0.05)
    selector.fit(X_ohe, y.as_matrix().ravel())
    X_ohe_selected = selector.transform(X_ohe)
    X_ohe_selected = pd.DataFrame(X_ohe_selected, columns=X_ohe_columns[selector.support_])
    logger.info('-----after feature selection------')
    logger.info(X_ohe_selected.shape)
    logger.info(X_ohe_selected.head(5))

    from sklearn.model_selection import train_test_split

    # Holdout
    X_train,X_test,y_train,y_test = train_test_split(X_ohe_selected,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=1,
                                                    shuffle=True)

    # 不均衡データ対応
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    rus = RandomUnderSampler(random_state=0)
    ros = RandomOverSampler(random_state=0)
    smt = SMOTE(random_state=0)

    logger.info('-----before resampling-----')
    logger.info(Counter(y_train))

    # resampling
    X_train_under, y_train_under = rus.fit_sample(X_train, y_train)
    X_train_over, y_train_over = ros.fit_sample(X_train, y_train)
    X_train_smt, y_train_smt = smt.fit_sample(X_train, y_train)

    logger.info('-----after resampling-----')
    logger.info('Random Under Sampler')
    logger.info(Counter(y_train_under))
    logger.info('Random Over Sampler')
    logger.info(Counter(y_train_over))
    logger.info('SMOTE')
    logger.info(Counter(y_train_smt))


    # ----モデル用データのmodeling----
    scores = {}
    for pipe_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train.as_matrix().ravel())
        logger.info(pipe_name + ': Fitting Done')
        scores[pipe_name, 'train'] = roc_auc_score(y_train.as_matrix().ravel(), pipeline.predict(X_train))
        scores[pipe_name, 'test'] = roc_auc_score(y_test.as_matrix().ravel(), pipeline.predict(X_test))
        joblib.dump(pipeline, '../model/'+ model_name + '_' + pipe_name  + '_' + '.pkl')

    # ----モデル用データ（under_sampling）のmodeling----
    scores_under = {}
    for pipe_name, pipeline in pipelines.items():
        pipeline.fit(X_train_under, y_train_under)
        logger.info(pipe_name + ': Fitting Done (under_sampling)')
        scores_under[pipe_name, 'train_under'] = roc_auc_score(y_train_under, pipeline.predict(X_train_under))
        scores_under[pipe_name, 'test'] = roc_auc_score(y_test.as_matrix().ravel(), pipeline.predict(X_test))
        joblib.dump(pipeline, '../model/'+ model_name + '_under_' + pipe_name  + '_' + '.pkl')

    # ----モデル用データ（over_sampling）のmodeling----
    scores_over = {}
    for pipe_name, pipeline in pipelines.items():
        pipeline.fit(X_train_over, y_train_over)
        logger.info(pipe_name + ': Fitting Done (over_sampling)')
        scores_over[pipe_name, 'train_over'] = roc_auc_score(y_train_over, pipeline.predict(X_train_over))
        scores_over[pipe_name, 'test'] = roc_auc_score(y_test.as_matrix().ravel(), pipeline.predict(X_test))
        joblib.dump(pipeline, '../model/'+ model_name + '_over_' + pipe_name  + '_' + '.pkl')

    logger.info('-----scores------')
    logger.info(pd.Series(scores).unstack())
    logger.info(pd.Series(scores_under).unstack())
    logger.info(pd.Series(scores_over).unstack())

    # preprocessing-4: preprocessing of a score data along with a model dataset（スコア用データの前処理）

    # load score data
    dfs = pd.read_csv('../data/'+ file_score + '.csv', header=0, dtype=my_dtype)
    # sort columns
    col = dfs.columns.tolist()
    col.remove('left')
    col.append('left')
    dfs2 = dfs.loc[:,col]
    IDs = dfs2.iloc[:,[0]] 
    Xs = dfs2.iloc[:,1:-1]

    # check the shape
    logger.info('---------------------------------')
    logger.info('Raw shape: (%i,%i)' %dfs.shape)
    logger.info('X shape: (%i,%i)' %Xs.shape)
    logger.info('---------------------------------')
    logger.info(Xs.dtypes)

    logger.info('-----dfs------')
    logger.info(dfs.head(5))
    logger.info('-----Xs------')
    logger.info(Xs.head(5))

    logger.info('-----check null exists------')
    logger.info(dfs.isnull().any())

    Xs_ohe = pd.get_dummies(Xs, dummy_na=True, columns=ohe_cols)
    cols_m = pd.DataFrame(None, columns=X_ohe_columns, dtype=float)

    logger.info('-----score data columns------')
    logger.info(cols_m)

    # consistent with columns set
    Xs_exp = pd.concat([cols_m, Xs_ohe])
    Xs_exp.loc[:,list(set(X_ohe_columns)-set(Xs_ohe.columns.values))] = Xs_exp.loc[:,list(set(X_ohe_columns)-set(Xs_ohe.columns.values))].fillna(0, axis=1) # モデルデータ（X_ohe）のみに存在する項目を0埋めしてスコアデータへセット

    logger.info('-----data of scores------')
    logger.info(Xs_exp.head(5))

    logger.info('-----feature list compare (before drop)------')
    cols_model = set(X_ohe_selected.columns.values)
    cols_score = set(Xs_ohe.columns.values)
    diff1 = cols_model - cols_score
    logger.info('-----columns that only exist in the model: %s ------' % diff1)
    diff2 = cols_score - cols_model
    logger.info('-----columns that only exist in the score: %s ------' % diff2)

    # score columns drop
    Xs_exp = Xs_exp.drop(list(set(Xs_ohe.columns.values)-set(X_ohe_selected.columns.values)), axis=1)

    logger.info('-----feature list compare (after drop)------')
    cols_model = set(X_ohe_selected.columns.values)
    cols_score = set(Xs_exp.columns.values)
    diff1 = cols_model - cols_score
    logger.info('-----columns that only exist in the model: %s ------' % diff1)
    diff2 = cols_score - cols_model
    logger.info('-----columns that only exist in the score: %s ------' % diff2)

    # re-order the score data columns （スコア用データの変数の並び順の制御）
    Xs_exp = Xs_exp.reindex_axis(X_ohe_columns, axis=1)

    logger.info('-----check null exists------')
    logger.info(Xs_exp.isnull().any())
    logger.info('-----before number of null------')
    logger.info(Xs_exp.isnull().sum().sum())
    Xs_exp = pd.DataFrame(imp.transform(Xs_exp), columns=X_ohe_columns)  # 数値変数の欠損値対応
    logger.info('-----after number of null------')
    logger.info(Xs_exp.isnull().sum().sum())
    Xs_exp_selected = Xs_exp.loc[:, X_ohe_columns[selector.support_]] # 学習済みのRFECVクラスのインスタンス（selector）を使用し選択された特徴量のインデックスを指定

    # スコア用データのscoring
    clf = joblib.load('../model/' + model_name + '_' + 'under_gb'  + '_' + '.pkl')
    score = pd.DataFrame(clf.predict_proba(Xs_exp_selected)[:,1], columns=['pred_score']) # スコア用データのscoring
    IDs.join(score).to_csv('../data/' +  model_name + '_' + file_score + '_with_pred.csv', index=False)

    # model profile（特徴量の重要度をcsv化）
    imp = pd.DataFrame([clf.named_steps['est'].feature_importances_], columns=X_ohe_columns[selector.support_])
    imp.T.to_csv('../data/' +  model_name + '_feature_importances.csv', index=True)
    logger.info('---- done! ----')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # SET PARAMETERS
    file_model = 'hr_analysis_train'
    file_score = 'hr_analysis_test'
    ohe_cols = ['sales', 'salary']
    model_name = 'CLASSIFIER_001'

    main()
