from dataset import dataset_numeric_classification
from util import kfold, random_replace_with_nan
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer, MissingIndicator
from sklearn.ensemble import ExtraTreesRegressor
from multiprocessing import Pool
import numpy as np


def run(datai, missing_rate):
    train_data, train_target, test_data, test_target = datai
    transformer = FeatureUnion([('1', IterativeImputer(ExtraTreesRegressor())), ('2', MissingIndicator(features='all'))])
    model = make_pipeline(StandardScaler(), transformer, SVC())
    model.fit(random_replace_with_nan(train_data, missing_rate), train_target)
    return accuracy_score(test_target, model.predict(random_replace_with_nan(test_data, missing_rate)))


def main():
    data_name = ['iris', 'wine',  'diabetes', 'ecoli', 'glass', 'seeds', 'yeast', 'thyroid-new', 'blood-transfusion-service-center']
    res = []
    missing_rate = 0.0
    for dn in data_name:
        data, target, att_dim, res_dim = dataset_numeric_classification(dn, 1)
        experiment_num, acc = 50, 0
        for en in range(experiment_num):
            p = Pool()
            r = [p.apply_async(run, args=(datai, missing_rate)) for datai in kfold(data, target, 5, 'numeric', en)]
            p.close(), p.join()
            acc += np.mean([_.get() for _ in r])/experiment_num
            print('now_round_%d' % en, end='\r')
        with open('res', 'a') as f:
            f.writelines(dn+str(acc)+'\n')
        res.append((dn, acc))
    print(res)


if __name__ == "__main__":
    main()

'''
SimpleImputer()
KNNImputer()
IterativeImputer()
IterativeImputer(ExtraTreesRegressor())

        iris    wine    pima    ecoli   glass   seeds   yeast   thyroid transfusion
mean                    
30%     88.42   94.55   72.85   77.95   57.16   89.08   50.37   91.06   76.27
50%     80.58   89.07   70.50   58.78   50.09   84.68   44.21   86.71   76.11
70%     67.69   77.50   68.02   59.09   44.39   74.96   38.35   81.09   76.10

knn
30%     89.77   95.77   72.64   76.84   59.49   89.55   49.47   92.00   76.26
50%     82.92   90.54   70.24   67.12   52.36   85.92   43.02   87.19   76.17
70%     68.86   79.68   67.79   57.48   45.18   76.84   37.41   81.51   76.13

mice
30%     91.84   94.05   73.01   77.76   60.46   90.25   50.98   91.41   76.35
50%     83.09   89.81   70.48   69.22   53.98   87.15   43.51   86.21   76.15
70%     67.79   77.76   67.70   59.31   45.73   76.85   38.18   80.65   76.11

missForest
30%     91.74   96.23   73.08   77.47   62.15   90.02   50.39   91.86   76.39
50%     82.93   91.29   70.24   67.23   55.73   86.97   42.83   87.69   76.22
70%     69.73   77.77   67.33   56.63   46.30   77.08   35.95   81.05   76.16

brb
30%     92.09   95.11   74.03   78.86   68.66   91.02   51.88   92.27
50%     86.24   91.89   70.57   71.96   64.49   87.73   46.49   89.16
70%
'''
