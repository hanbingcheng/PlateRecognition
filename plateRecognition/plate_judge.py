'''
Created on 2018/02/03

@author: teikanhei
'''

'''
Created on 2018/01/27

@author: hanbing.cheng
'''
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from common import Configuration
from plateRecognition.features import CreateFeatures
from sklearn.metrics import accuracy_score

class PlateSVMClassfier:
    
    def __init__(self):
        conf = Configuration.get_datamodel_storage_path()
        self.model_path = conf["Plate_SVM_RBF"]
        
        if not os.path.isfile(self.model_path):
            self.train_model()
        self.model = joblib.load(self.model_path)
    
    # Read the features stored in the disk
    def train_model(self):
        cf = CreateFeatures()
        cf.create_training_feature(store='Yes')
        #features = np.genfromtxt(conf['Data_feature_dir'], dtype=float, delimiter=',')
        #labels = np.genfromtxt(conf['Class_labels_dir'], dtype=float, delimiter=',')
        features = cf.features
        labels = cf.labels
        print("shape of features:")
        print (features.shape)
        # use svm to train
        #sigma=4
        #max_iter=5000
        #alpha=0.003
        #c=10
        #Models(type = 'rbf').fit(features, labels)
    
        # using gridsearch to get best parameters
        '''
        model_to_set = SVC()
        parameters = {
            "C": [0.1,0.5,1.0,1.5, 2.0, 3.0, 5.0, 10.0],
            "kernel": ["rbf"],
            "degree":[1, 2, 3, 4],
            "gamma": [0.001, 0.01, 0.02, 0.1]
        }
    
        clf = GridSearchCV(model_to_set, param_grid=parameters)
        clf.fit(features, labels)
        print("best_params_")
        print(clf.best_params_)
        print("grid_scores_")
        print(clf.grid_scores_)
        '''
        C = 9.
        kernel = 'rbf'
        gamma = 0.7
        degree = 1
        clf = SVC(C = C, kernel = kernel, gamma = gamma, degree = degree)
        clf.fit(features, labels)
        
        prediction = clf.predict(features)
        score = accuracy_score(labels, prediction)
        print ('accuracy_score: {:.5f}'.format(score))
        # save model
        joblib.dump(clf, self.model_path)
        
        print ("train finished.")
    
    # #==============================================================================
    # # 3: Use the patameters and Operate on cross validation dataset
    # #==============================================================================
    
    '''
    def valid(inp_path, model=None):
        feature_valid, _ = CrtFeatures().create_features(inp_path)
        feature_matrix_valid = np.array(feature_valid, dtype="float64")
        if model=='rbf':
            prediction_dict = Eval().test_using_model_rbf(feature_matrix_valid)
        elif model=='self':
            prediction_dict = Eval().test_using_model_self(feature_matrix_valid)
        else:
            print ('You should specify the model in which you would wanna crossvalidate your data')
        return prediction_dict
    
    def run_cross_valid():
        valid_path_LP = [conf['Valid_data1_dir']]
        valid_path_non_LP = [conf['Valid_data2_dir']]
        for no, path in enumerate([valid_path_LP, valid_path_non_LP]):
            print ('Running classification no validation file %s:  '%path)
            prediction_dict = valid(inp_path=path, model='rbf')
            for model, pred in prediction_dict.items():
                if no==0:
                    labels_valid = np.ones(len(pred))
                elif no==1:
                    labels_valid = np.zeros(len(pred))
                accuracy=accuracy_score(labels_valid, pred)
                print ('The accuracy of model %s is: '%model, accuracy)
    
    '''
        
        
       
    def predict(self, test_features):
        test_features = test_features.reshape(1, 648)
        predition = self.model.predict(test_features)

        return predition
    
    def isPlate(self, test_features):
        prediction = self.predict(test_features)
       
        if (prediction == 1):
            return True
        else:
            return False
        
        
            
if __name__ == "__main__":
    
    clf = PlateSVMClassfier()
    clf.train_model()
            