import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from collections import Counter
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.feature_selection import SelectKBest,SelectFromModel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict

def transform_data(train_location, cat):
    
    #Loads data
    titanic_df = pd.read_csv(train_location, header=0 )
    
    titanic_df.fillna(0)
    
    #feature creation
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty":5, "Officer": 6}
    titanic_df ['Deck'] = titanic_df ['Cabin'].astype(str).str[0]
    titanic_df ['Cabin Number'] = titanic_df ['Cabin'].astype(str).str[1]
    #titanic_df ['Cabin Number'] = titanic_df ['Cabin Number'].replace(['a'], '')
    titanic_df ['Title'] = titanic_df ['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    titanic_df ['Title'] = titanic_df ['Title'].replace(['Lady', 'Countess', 'Dona'],'Royalty')
    titanic_df ['Title'] = titanic_df ['Title'].replace(['Mme'], 'Mrs')
    titanic_df ['Title'] = titanic_df ['Title'].replace(['Mlle','Ms'], 'Miss')
    titanic_df ['Title'] = titanic_df ['Title'].replace(['Capt', 'Col', 'Major','Rev'], 'Officer')
    titanic_df ['Title'] = titanic_df ['Title'].replace(['Jonkheer', 'Don','Sir'], 'Royalty')
    titanic_df .loc[(titanic_df .Sex == 'male')   & (titanic_df .Title == 'Dr'),'Title'] = 'Mr'
    titanic_df .loc[(titanic_df .Sex == 'female') & (titanic_df .Title == 'Dr'),'Title'] = 'Mrs'
    titanic_df ['Title'] = titanic_df ['Title'].map(title_mapping)
    titanic_df ['Family'] = titanic_df ['SibSp'] + titanic_df ['Parch']
    titanic_df ['Age*Class'] = titanic_df .Age * titanic_df .Pclass
    titanic_df = titanic_df.replace(['None'], '0')
    
    
    titanic_df ['Sex'] = titanic_df ['Sex'].map( {'female': 1, 'male': 0} ).astype(float)
    titanic_df ['Embarked'] = titanic_df ['Embarked'].map( {'C': 1, 'Q': 2, 'S': 3} ).astype(float)
    titanic_df ['Deck'] = titanic_df ['Deck'].map( {'A': 0, 'B': 1, 'C': 2,'D': 3, 'E': 4, 'F': 5,'G': 6, 'T': 7} ).astype(float)
    titanic_df ['Cabin Number'] = titanic_df ['Cabin Number'].map( {'A': 0, 'B': 1, 'C': 2,'D': 3, 'E': 4, 'F': 5,'G': 6, 'T': 7} ).astype(float)
    
    #for c in titanic_df:
    #    if str(titanic_df[c].dtype) in ('object', 'string_', 'unicode_'):
    #        titanic_df[c].fillna(value='', inplace=True)
    
    
    
    titanic_df  = titanic_df .drop(['Name'], axis=1)
    titanic_df  = titanic_df .drop(['Ticket'], axis=1)
    titanic_df  = titanic_df .drop(['PassengerId'], axis=1)
    titanic_df  = titanic_df .drop(['Cabin'], axis=1)        
    #titanic_df  = titanic_df .drop(['Cabin Number'], axis=1)
    #titanic_df  = titanic_df .drop(['Deck'], axis=1)
        
    # Pull out features for future use
    features_names = titanic_df.columns.tolist()

    print ("Column names:")
    print (features_names)
    
    
    titanic_df = pd.get_dummies(titanic_df)
    
    print(titanic_df.head())  
    
    if (cat==1):
        X = titanic_df.drop("Survived", axis=1)
        Y = titanic_df["Survived"]
        
        X = X.as_matrix().astype(np.int)
        Y = Y.as_matrix().astype(np.int)
        
        return X, Y
    else:
        X  = titanic_df
        
        X = X.as_matrix().astype(np.int)
        
        return X
        

train_location = "C:/Users/Oliver Crosbie Higgs/Documents/personal projects/train.csv"
test_location = "C:/Users/Oliver Crosbie Higgs/Documents/personal projects/test.csv"

X_train, Y_train = transform_data(train_location,1)
X_test = transform_data(test_location,0)

forest = RandomForestClassifier(n_estimators=250,random_state=0)
model1a = DTC(max_depth=10)

classifier = OneVsRestClassifier(model1a)
y_score = classifier.fit(X_train, Y_train).predict(X_test.astype(int))

np.savetxt("y_score.csv", y_score, delimiter=",")