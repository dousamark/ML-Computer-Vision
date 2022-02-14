import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
import pickle
import lzma

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay)

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def onehot(data):
    oneHotArray=[]
    for i in range(len(data)):
        oneTarget =np.zeros(100)
        oneTarget[data[i]-1]=1
        oneHotArray.append(oneTarget)
    return oneHotArray

def separate(data, target, taskOne, fine):
    choosen = [1,19]
    selected_data=[]
    selected_targets=[]
    for i in range(len(target)):
        if(target[i] == choosen[0] or (target[i] == choosen[1] and taskOne)):
            selected_data.append(data[i])
            if(not taskOne):
                selected_targets.append(fine[i]) 
            else:   
                selected_targets.append(target[i])
    return selected_data,selected_targets

def showConfusionMatrix(test_target,predictions):
    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(confusion_matrix(test_target, predictions), interpolation ='nearest')
    figure.colorbar(caxes)
    plt.show()

def showPrecisionRecallCurve(test_target, predictions):
    normalized_target = []
    normalized_pred = []
    for i in range(len(test_target)):
        if(test_target[i]==1):
            normalized_target.append(1)
        else:
            normalized_target.append(0)
        
        if(predictions[i]==1):
            normalized_pred.append(1)
        else:
            normalized_pred.append(0)
    
    precision, recall, _ = precision_recall_curve(normalized_target, normalized_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()

def crossValidationForKNearest(train_data,train_target):
    Ks = np.arange(1,15)
    best_k = 0
    best_score=0
    #5fold splits size to 0.8 and 0.2
    devel_size=0.2
    for k in Ks:
        scores=[]
        #5fold
        for i in range(5):
            X_train, X_devel, y_train, y_devel = sklearn.model_selection.train_test_split(train_data, train_target, test_size=devel_size, random_state=i)
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train,y_train)
            predictions = model.predict(X_devel)
            scores.append(accuracy_score(y_devel, predictions, normalize=True, sample_weight=None))

        averagedScore= np.mean(scores)
        if(averagedScore>best_score):
            best_score = averagedScore
            best_k = k
    return best_k

def crossValidationForDT(train_data,train_target):
    splitSize = np.arange(3,15)
    best_size = 0
    best_score=0
    #5fold splits size to 0.8 and 0.2
    devel_size=0.2
    for size in splitSize:
        scores=[]
        #5fold
        for i in range(5):
            X_train, X_devel, y_train, y_devel = sklearn.model_selection.train_test_split(train_data, train_target, test_size=devel_size, random_state=i)
            model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=size)
            model = model.fit(train_data, train_target)
            predictions = model.predict(X_devel)
            scores.append(accuracy_score(y_devel, predictions, normalize=True, sample_weight=None))

        averagedScore= np.mean(scores)
        if(averagedScore>best_score):
            best_score = averagedScore
            best_size = size
    return best_size

def crossValidationForMLP(train_data,train_target):
    lambdas = np.geomspace(0.01, 10, num=50)
    best_lambda = 0
    best_score=0
    #5fold splits size to 0.8 and 0.2
    devel_size=0.2
    
    for lamb in lambdas:
        scores=[]
        #5fold
        for i in range(5):
            X_train, X_devel, y_train, y_devel = sklearn.model_selection.train_test_split(train_data, train_target, test_size=devel_size, random_state=i)
            model = MLPClassifier(hidden_layer_sizes=(500, 200), activation='logistic', solver='adam', alpha=best_lambda,
        batch_size='auto', learning_rate='adaptive', max_iter=20, random_state=42)
            model = model.fit(X_train, y_train)
            predictions = model.predict(X_devel)
            scores.append(accuracy_score(y_devel, predictions, normalize=True, sample_weight=None))

        averagedScore= np.mean(scores)
        if(averagedScore>best_score):
            best_score = averagedScore
            best_lambda = lamb
    return best_lambda

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#download dataset
dataset_train = unpickle("train")
train_data = dataset_train.get(b"data")
train_target = dataset_train.get(b"coarse_labels")
train_target_fine = dataset_train.get(b"fine_labels")

#separating set for desired training datasets
taskOne=True
if(taskOne):
    separated_train_data,separated_train_target = separate(train_data, train_target, True,None)
    pca = PCA(768) #16*16*3 = 768
else:
    separated_train_data,separated_train_target = separate(train_data, train_target, False,train_target_fine)
    pca = PCA(432) #12*12*3 = 768

#reducing the number of dimensions
fitted_dataset = pca.fit_transform(separated_train_data,separated_train_target)

#classification models
def predictMLP(train_data,train_target,test_data,test_target):
    best_lambda = crossValidationForMLP(train_data,train_target)
    model = MLPClassifier(hidden_layer_sizes=(1000, 300), activation='logistic', solver='adam', alpha=best_lambda,
        batch_size='auto', learning_rate='adaptive', max_iter=100, random_state=42, verbose=True)
    model = model.fit(train_data, train_target)
    predictions = model.predict(test_data)
    
    if(taskOne):
        print("Accuracy Neural Network Task One",end=" : ")
        modelName = "mlp_task1.model"
    else:
        print("Accuracy Neural Network Task Two",end=" : ")
        modelName = "mlp_task2.model"
    print(str(accuracy_score(test_target, predictions, normalize=True, sample_weight=None)*1000%1000/10)+"%")
    
    #saving model
    with lzma.open(modelName, "wb") as model_file:
            pickle.dump(model, model_file)

    #showConfusionMatrix(test_target,predictions)
    #showPrecisionRecallCurve(test_target, predictions)

def predictKNearest(train_data,train_target,test_data,test_target):
    Kparameter = crossValidationForKNearest(train_data,train_target)
    model = KNeighborsClassifier(n_neighbors=Kparameter)
    model.fit(train_data,train_target)
    predictions = model.predict(test_data)
    prob_predictions = model.predict_proba(test_data)
    
    if(taskOne):
        print("Accuracy K Nearest Task One",end=" : ")
        modelName="KNearest_task1.model"
    else:
        print("Accuracy K Nearest Task Two",end=" : ")
        modelName="KNearest_task2.model"
    print(str(accuracy_score(test_target, predictions, normalize=True, sample_weight=None)*1000%1000/10)+"%")

    #saving model
    with lzma.open(modelName, "wb") as model_file:
            pickle.dump(model, model_file)

    #showConfusionMatrix(test_target,predictions)
    #showPrecisionRecallCurve(test_target, predictions)
    

def predictDecTree(train_data,train_target,test_data,test_target):
    minSamplesToSplit = crossValidationForDT(train_data,train_target)
    model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=minSamplesToSplit)
    model = model.fit(train_data, train_target)
    predictions = model.predict(test_data)
    
    
    if(taskOne):
        print("Accuracy Decision Tree Task One",end=" : ")
        modelName= "decisionTree_task1.model"
    else:
        modelName= "decisionTree_task2.model"
        print("Accuracy Decision Tree Task Two",end=" : ")
    print(str(accuracy_score(test_target, predictions, normalize=True, sample_weight=None)*1000%1000/10)+"%")
    
    #saving model
    with lzma.open(modelName, "wb") as model_file:
            pickle.dump(model, model_file)

    #showConfusionMatrix(test_target,predictions)
    #showPrecisionRecallCurve(test_target, predictions)

#preparing data for testing
dataset_test = unpickle("test")
test_data = dataset_test.get(b"data")
test_target = dataset_test.get(b"coarse_labels")
test_target_fine = dataset_test.get(b"fine_labels")

if(taskOne):
    separated_test_data,separated_test_target = separate(test_data,test_target, True,None)
else:
    separated_test_data,separated_test_target = separate(test_data,test_target, False,test_target_fine)

fitted_test_data = pca.fit_transform(separated_test_data,separated_test_target)

predictMLP(fitted_dataset,separated_train_target,fitted_test_data,separated_test_target)
predictKNearest(fitted_dataset,separated_train_target,fitted_test_data,separated_test_target)
predictDecTree(fitted_dataset,separated_train_target,fitted_test_data,separated_test_target)