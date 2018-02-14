'''
Created on 2018/01/27

@author: hanbing.cheng
'''
from sklearn.svm import SVC
from sklearn.externals import joblib
#from sklearn.grid_search import GridSearchCV
#from sklearn.multiclass import OneVsRestClassifier
from common import  configuration

class CharSvmModel:
    
    def __init__(self):
        conf = configuration.get_datamodel_storage_path()
        self.model_path   =  conf["Char_SVM_RBF"]
        
    def fit(self, train_data, train_label):
        # using gridsearch to get best parameters
        #model_to_set = SVC()
        #parameters = {
        #    "C": [0.1,0.5,1.0,1.5, 2.0, 3.0],
        #    "kernel": ["rbf"],
        #    "degree":[1, 2, 3, 4],
        #    "gamma": [0.001, 0.01, 0.02, 0.1]
        #}
    
        #clf = GridSearchCV(model_to_set, param_grid=parameters)
        #print("best_params_")
        #print(clf.best_params_)
        #print("grid_scores_")
        #print(clf.grid_scores_)
        
        C = 3.
        kernel = 'rbf'
        gamma = 0.001
        degree = 1
        clf = SVC(C = C, kernel = kernel, gamma = gamma, degree = degree)
        clf.fit(train_data, train_label)
        
        # save model
        joblib.dump(clf, self.model_path)
        print ("train finished.")
       
    def predict(self, test_data):
        clf = joblib.load(self.model_path)
        predition = clf.predict(test_data) 
        return predition