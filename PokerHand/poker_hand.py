import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
numFolds = 11
##Attributes Information
'''     1) S1 "Suit of card #1" 
        Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 
        
        2) C1 "Rank of card #1" 
        Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 
        
        3) S2 "Suit of card #2" 
        Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 
        
        4) C2 "Rank of card #2" 
        Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 
        
        5) S3 "Suit of card #3" 
        Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 
        
        6) C3 "Rank of card #3" 
        Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 
        
        7) S4 "Suit of card #4" 
        Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 
        
        8) C4 "Rank of card #4" 
        Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 
        
        9) S5 "Suit of card #5" 
        Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 
        
        10) C5 "Rank of card 5" 
        Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 
        
        11) CLASS "Poker Hand" 
        Ordinal (0-9) '''
def readData():
    features = ['Scard1', 'Rcard1', 'Scard2', 'Rcard2', 'Scard3', 'Rcard3', 'Scard4', 'Rcard4', 'Scard5','Rcard5','PokerHand' ]
    c_df = pd.read_csv('pokertraining.csv', names = features)
    return c_df

def splitData(c_df):
    kf = KFold(n_splits=numFolds, shuffle=True)
    return kf

def cal(predictions, true):
    l = len(np.unique(true))
    falseP = 0
    trueP = 0
    for (x,y) in zip(predictions, true):
        if x!=y:
            falseP+=1
        else:
            trueP+=1
    return((falseP/(1.0*l)), trueP/(1.0*l))

def main():
    c_df = readData()
    kf = splitData(c_df)
    X = c_df.drop('PokerHand', axis=1)
    Y = c_df['PokerHand']
    total_score = 0;
    total_f1_score = 0;
    y_pred = []
    y_true = []
    points = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        gnb = GaussianNB().fit(X_train, Y_train)
        gnb_predictions = gnb.predict(X_test)
        y_pred.extend(gnb_predictions)
        y_true.extend(list(Y_test))
        acc_score = accuracy_score(Y_test, gnb_predictions)
        total_score+=acc_score;
        score = f1_score(Y_test, gnb_predictions, average='weighted', labels=np.unique(gnb_predictions))
        total_f1_score += score
        falseP, trueP = cal(gnb_predictions, Y_test)
        points.append((falseP, trueP))
    Accuracy = total_score/numFolds
    F1_Score = total_f1_score/numFolds
    CM = confusion_matrix(y_true, y_pred)
    print("Output:")
    print(CM)
    print('Accuracy: {0}%'.format(Accuracy*100))
    print('F1_Score: {0}'.format(F1_Score))
    
    for x in points:
        falseP = x[0]
        trueP = x[1]
        plt.scatter(falseP, trueP)
    plt.xlabel('false Pos')
    plt.ylabel('true Pos')
    plt.plot(x,x)
    plt.show()    

main()

