
# Predicting Judicial Judgement of Lawsuits using Natural Language Processing and Machine Learning

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import metrics
from sklearn import svm
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


# This is the path where we have extracted the dataset.

DATASET_PATH = './dataset/'


def get_labels(path):
# This is the path to the case files e.g. dataset/ArticleX/cases_aX.csv

    try:   
        data = pd.read_csv(path, header=None).as_matrix()
        case_names = data[:,0]
        labels = data[:,1]

    except FileNotFoundError: 
        messagebox.showinfo("Error", "Choose an Article")

    return labels, case_names


# We first extract the labels (violation/no-violation) of the cases in the dataset.

def main(a_selection,f_selection):
    ARTICLE_NUM = str(a_selection)
    DATA_PATH = DATASET_PATH+'Article'+ARTICLE_NUM+'/'

    # load labels (violation/no-violation) and case names
    Y, case_names = get_labels(path=DATA_PATH+'/cases_a'+ARTICLE_NUM+'.csv')


    # We then load the features extracted from the cases for any one Article.
    # - __FEAT_TOPICS:__ Topics for each article by clustering together N-grams that are semantically similar.
    # - __FEAT_PROCEDURE:__ N-gram features for the Procedure part of the case
    # - __FEAT_CIRCUMSTANCES:__ N-gram features for the Circumstances part of the case
    # - __FEAT_RELEVANTLAW:__ N-gram features for the Relevant Law
    # - __FEAT_LAW:__ N-gram features for the Law
    # - __FEAT_FULL:__ N-gram features from all the parts above
    # - __FEAT_FACTS:__ Combination of N-gram features of Circumstances and Relevant Law
    # - __FEAT_TOPICS_CIRCUMSTANCES:__ Combination of topics and Circumstances N-grams


    # Available features
    FEAT_TOPICS = 'topics'+ARTICLE_NUM+'.csv'
    FEAT_PROCEDURE = 'ngrams_a'+ARTICLE_NUM+'_procedure.csv'
    FEAT_CIRCUMSTANCES = 'ngrams_a'+ARTICLE_NUM+'_circumstances.csv'
    FEAT_RELEVANTLAW = 'ngrams_a'+ARTICLE_NUM+'_relevantLaw.csv'
    FEAT_LAW = 'ngrams_a'+ARTICLE_NUM+'_law.csv'
    FEAT_FULL = 'ngrams_a'+ARTICLE_NUM+'_full.csv'
    FEAT_FACTS = 'circumstances+relevantLaw'
    FEAT_TOPICS_CIRCUMSTANCES = 'topics+circumstances'

    # choose one of the features above
    if f_selection==1:
        FEAT=FEAT_TOPICS
        FEAT_Name="Topics"
        
    elif f_selection==2:
        FEAT=FEAT_PROCEDURE
        FEAT_Name="Procedure"
        
    elif f_selection==3:
        FEAT=FEAT_CIRCUMSTANCES
        FEAT_Name="Circumstances"
        
    elif f_selection==4:
        FEAT=FEAT_RELEVANTLAW
        FEAT_Name="Relevent Law"
        
    elif f_selection==5:
        FEAT=FEAT_LAW
        FEAT_Name="Law"
        
    elif f_selection==6:
        FEAT=FEAT_FULL
        FEAT_Name="Full"
        
    elif f_selection==7:
        FEAT=FEAT_FACTS
        FEAT_Name="Facts"
        
    elif f_selection==8:
        FEAT=FEAT_TOPICS_CIRCUMSTANCES
        FEAT_Name="Topics & Circum"

    # load features
    try:    
        if FEAT == FEAT_TOPICS_CIRCUMSTANCES:
            X_topics = np.loadtxt(open(DATA_PATH+FEAT_TOPICS, 'r', encoding='latin-1'), delimiter='\t', dtype=float)
            X_circ = np.loadtxt(open(DATA_PATH+FEAT_CIRCUMSTANCES, 'r', encoding='latin-1'), delimiter=',', dtype=float)
            X = np.concatenate((X_topics, X_circ), axis=1) #ValueError(if average) : shapes (250,30) (250,2000)
        elif FEAT == FEAT_FACTS:
            X_relev = np.loadtxt(open(DATA_PATH+FEAT_RELEVANTLAW, 'r', encoding='latin-1'), delimiter=',', dtype=float)
            X_circ = np.loadtxt(open(DATA_PATH+FEAT_CIRCUMSTANCES, 'r', encoding='latin-1'), delimiter=',', dtype=float)
            X = (X_relev + X_circ)/2.0
        else:
            try:
                X = np.loadtxt(open(DATA_PATH+FEAT, 'r', encoding='latin-1'), delimiter=',', dtype=float)
            except:
                X = np.loadtxt(open(DATA_PATH+FEAT, 'r', encoding='latin-1'), delimiter='\t', dtype=float)

    except UnboundLocalError:
        messagebox.showinfo("Error", "Choose a Feature")

    logging.info('Starting CV')
    count = 1
    mean_acc = 0.0

    scores = []

    # load cv fold, train and test indices
    train_inds = pd.DataFrame.from_csv('train_folds_article'+ARTICLE_NUM+'.csv',header=None, index_col=None).as_matrix().astype(float)
    test_inds = pd.DataFrame.from_csv('test_folds_article'+ARTICLE_NUM+'.csv',header=None, index_col=None).as_matrix().astype(float)
    #print(train_inds)
    #print(test_inds)

    for train_index, test_index in zip(train_inds, test_inds):
        
        logging.info('Fold'+str(count))
        
        train_index = train_index[~np.isnan(train_index)].astype(int)
        test_index = test_index[~np.isnan(test_index)].astype(int)
        #print(train_index)
        #print(test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        #print("X train",X_train)
        #print("X test",X_test)
        #print("Y train",Y_train)
        #print("Y test",Y_test)
        
        clf = svm.SVC(C=10e+1,kernel='linear')
        clf.fit(X_train,Y_train)

        count+=1

        predicted = clf.predict(X_test)
        print(metrics.classification_report(Y_test, predicted, target_names=['violation','no_violation'], digits=3))
        print(Y_test,predicted)
        acc = np.mean(predicted == Y_test)
        print('Accuracy:\t'+str(round(acc, 4)))
        mean_acc+=acc
        scores.append(acc)
    e1.delete(0, END)
    e1.insert(0,str(round(mean_acc/10.0, 4)))
    e2.delete(0, END)
    e2.insert(0,str(round(np.std(scores),4)))
    print('\nMean Accuracy:\t'+str(round(mean_acc/10.0,4)))
    print('Standard Dev:\t'+str(round(np.std(scores),4)))

    if ARTICLE_NUM=="3":
        if round(mean_acc/10.0, 3)>=0.75:
            summary.set("The Article 3 falls in the category of 'Freedom from torture'. Mean accuracy\n evaluated by the feature : "+FEAT_Name+" for prediction is : "+str(round(mean_acc*10.0,2))+"%. As the accuracy is \nmore than our threshold accuracy(75%), the model is reliable." )
        else:
            summary.set("The Article 3 falls in the category of 'Freedom from torture'. Mean accuracy evaluated by\n the feature : "+FEAT_Name+" for prediction is : "+str(round(mean_acc*10.0,2))+"%. As the accuracy is less than our\n threshold accuracy(75%), more dataset for training is needed to increase the accuracy." )
            
    elif ARTICLE_NUM=="6":
        if round(mean_acc/10.0, 3)>=0.75:
            summary.set("The Article 6 falls in the category of 'Right to a Fair Trial & Public Hearing'. Mean\n accuracy evaluated by the feature : "+FEAT_Name+" for prediction is : "+str(round(mean_acc*10.0,2))+"%. As\n the accuracy is more than our threshold accuracy(75%), the model is reliable." )
        else:
            summary.set("The Article 6 falls in the category of 'Right to a Fair Trial & Public Hearing'. Mean\n accuracy evaluated by the feature : "+FEAT_Name+" for prediction is : "+str(round(mean_acc*10.0,2))+"%. As the accuracy \nis less than our threshold accuracy(75%), more dataset for training is needed\n to increase the accuracy." )

            
    elif ARTICLE_NUM=="8":
        if round(mean_acc/10.0, 3)>=0.75:
            summary.set("The Article 8 falls in the category of 'Respect for private and family life'. Mean accuracy\n evaluated by the feature : "+FEAT_Name+" for prediction is : "+str(round(mean_acc*10.0,2))+"%. As the\n accuracy is more than our threshold accuracy(75%), the model is reliable." )
        else:
            summary.set("The Article 8 falls in the category of 'Respect for private and family life'. Mean accuracy\n evaluated by the feature : "+FEAT_Name+" for prediction is : "+str(round(mean_acc*10.0,2))+"%. As the accuracy is less\n than our threshold accuracy(75%), more dataset for training is needed to increase\n the accuracy." )

            
def compute():
    a_selection=v1.get()
    f_selection=v.get()
    main(a_selection,f_selection)
    #print(v1.get())
    #print(v.get())
    v1.set(0)
    v.set(0)

#----------------Graphical user interface------------------
    
from tkinter import *
from ttkthemes import themed_tk as tk
from tkinter import ttk,messagebox

root = tk.ThemedTk()
root.get_themes()
root.set_theme("arc")

root.title("Predicting Judicial Judgement using NLP and ML")
#root.geometry('600x600')

window_width=650
window_height=600
screenwidth=root.winfo_screenwidth()
screenheight=root.winfo_screenheight()

xcoordinate=(screenwidth/2)-(window_width/2)
ycoordinate=(screenheight/2)-(window_height/2)
root.geometry("%dx%d+%d+%d" % (window_width,window_height,xcoordinate, ycoordinate))
root.configure(background="#F5F6F7")
root.resizable(False, False)

v1 = IntVar()#Article Selection
v = IntVar()#Feature Selection

ttk.Label(root,text="Choose an Article :",justify = LEFT).grid(row=0,column=0,padx=10,pady = 30,sticky=W)

ttk.Radiobutton(root,text="Article 3",variable=v1,value=3).grid(row=1,column=1,padx = 0,sticky=W)
ttk.Radiobutton(root,text="Article 6",variable=v1,value=6).grid(row=1,column=2,padx = 0,sticky=W)
ttk.Radiobutton(root,text="Article 8",variable=v1,value=8).grid(row=1,column=3,padx = 0,sticky=W)

ttk.Label(root, text="Features Selection :").grid(row=3,pady=30,padx=10,sticky=W)

ttk.Radiobutton(root, text="Topics", variable=v,value=1).grid(row=4,column=0,padx=10,sticky=W)
ttk.Radiobutton(root, text="Procedure", variable=v,value=2).grid(row=4,column=1,padx=10,sticky=W)
ttk.Radiobutton(root, text="Circumstances", variable=v,value=3).grid(row=4,column=2,padx=10)
ttk.Radiobutton(root, text="Relevant Law", variable=v,value=4).grid(row=4,column=3,padx=10,sticky=W)
ttk.Radiobutton(root, text="Law", variable=v,value=5).grid(row=5,column=0,padx=10,pady=10,sticky=W)
ttk.Radiobutton(root, text="Full", variable=v,value=6).grid(row=5,column=1,padx=10,pady=10,sticky=W)
ttk.Radiobutton(root, text="Facts", variable=v,value=7).grid(row=5,column=2,padx=10,pady=10,sticky=W)
ttk.Radiobutton(root, text="Topics Circumstances", variable=v,value=8).grid(row=5,column=3,padx=10,pady=10,sticky=W)

ttk.Label(root,text="Mean Accuracy :").grid(row=7,column=0,pady=20,padx=10,sticky=W)
e1 = Entry(root,fg="#29524A")
e1.grid(row=7,column=1)

ttk.Label(root,text="Standard Deviation :").grid(row=8,column=0,padx=10,pady=20,sticky=W)
e2 = Entry(root,fg="#29524A")
e2.grid(row=8,column=1)

summary=StringVar()
ttk.Label(root,text="Summary :").grid(row=9,column=0,pady=20,padx=10,sticky=W)
ttk.Label(root,textvariable=summary).grid(row=9,column=1,pady=30, columnspan=5,sticky=W)
ttk.Button(root, text='Execute', command=compute).grid(row=10,column=1,columnspan=2,pady=30)

root.mainloop()
