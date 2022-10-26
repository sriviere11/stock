
#from VBTT_IO.IO import *
#from VBTT_SP500.SP500 import *
#from VBTT_Features.Features import *
#from VBTT_Algo.Algo import *
#from VBTT_Algo.GenerateModel import *
#from VBTT_Predict.Predict import *

#functions to fetch and process SP500 data

def read_create_write_SP500(SP500_tickers,filename_json):
    # This create the complet master list of all tickers in SP500, their sectors, their industries
    # Check if SP500 json file exist first otherwise process SP500

    file_exists = os.path.exists(filename_json)

    if file_exists:
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
    else:
        SP500_list = read_write_SP500(SP500_tickers,filename_json)  # this will extract

    return SP500_list


def get_ticker_sector(ticker):
    # version 2
    # prerequisites is to import Yahoo_finance.stock_info
    # probleme avec cette methode c'est que industry est ligne 18 ou 19 *
    # prerequisites 2 is to import pandas as pd; import numpy as np
    df = si.get_company_info(ticker)  # a utiliser pour trouver le secteur
    df = df.reset_index()
    sector = df.loc[df['Breakdown'].isin(['sector'])]  # from bamboolib to extract sector
    sector = sector.iloc[0, 1]  # this is to extract just the value
    industry = df.loc[df['Breakdown'].isin(['industry'])]
    industry = industry.iloc[0, 1]
    ticker_sector = []
    ticker_sector.append([ticker, sector, industry])
    return ticker_sector[0]


def read_write_SP500(tickers_list,filename_json):
    # Initialisation of SP500 data - find sector, industry for all tickers in SP500

    SP500_list = []
    for ticker in tickers_list:
        SP500_list.append(get_ticker_sector(ticker))
        if len(SP500_list) % 50 == 0:
            time.sleep(
                30)  # for some reason processing SP500 one shot is failing. so sleep of 20 seconds for each 50 tickers
    SP500_list = np.array(SP500_list)
    # Save in a file to reduce processing time next time
    write_list(SP500_list.tolist(), filename_json)  # this function work if it is a list
    return SP500_list  # this returns the array, not the list


def get_all_tickers_sector(list, sector):
    # prerequisites list is an array of ticker, sector and industry
    sub_list_tickers = np.array(list)
    fltr = np.asarray([sector])
    result = sub_list_tickers[np.in1d(sub_list_tickers[:, 1], fltr)]
    return result[:, 0:1]
    # on veut extraire les tickers donc toutes les rangees (0:0) et la colonne 0 donc (0:1)


def get_all_tickers_industry(list, industry):
    # prerequisites list is an array of ticker, sector and industry
    sub_list_tickers = np.array(list)
    fltr = np.asarray([industry])
    result = sub_list_tickers[np.in1d(sub_list_tickers[:, 2], fltr)]
    return result[:, 0:1]
    # on veut extraire les tickers donc toutes les rangees (0:0) et la colonne 0 donc (0:1)


from datetime import datetime,timedelta
import matplotlib
import logging
from matplotlib import pyplot as plt
from yahoo_fin import stock_info as si
from yahoo_fin import *
import pandas as pd
import pandas
import seaborn as sns
import numpy as np

#import bamboolib as bam
import json
import os.path
import math
import joblib
import os


from sklearn.metrics import balanced_accuracy_score


#### Preprocessing - generate matrix of features for sector or industry
def Model_Train_Save(ticker_list_for_models,years,lags,additional_data, nb_predict_days):
    # Functions for model training and algorythm data and import


    # import the regressors
    from sklearn import linear_model, svm
    from sklearn.tree import DecisionTreeRegressor
    # MODEL = linear_model.LinearRegression()
    # MODEL = svm.SVR()
    MODEL = DecisionTreeRegressor()

    # import balanced_accuracy_score
    from sklearn.metrics import balanced_accuracy_score

    #### Initialisation of variables and data
    # the timeframe for training and test sets and predict
    days = 360 * years  # Nunber of days in the model
    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(days,
                                                                                                                lags,
                                                                                                                nb_predict_days)
    SP500_tickers = si.tickers_sp500()  # get list of tickers

    # print(f"SP500_tickers = {SP500_tickers}")

    #additional_data = read_config_file()
    # read master list of ticker, sector, industries or if not found, create it and save it#
    SP500_list = read_create_write_SP500(SP500_tickers, "SP500.json")

    ### validation if we have everything
    #print(f"Input ->years {years}")
    #print(f"Input ->ticker list {ticker_list_for_models}")
    #print(f"Input->lags {lags}")
    #print(f"Input ->predict days {nb_predict_days}")
    #print(f"Period->yesterday {yesterday}")
    #print(f"period->train_date_start {train_date_start}")
    #print(f"Period->train_date_last {train_date_last}")
    #print(f"Period->test_date_start {test_date_start}")
    #print(f"Period->test_date_last {test_date_last}")

    #print(f"This is the tickers for our model {ticker_list_for_models}")
    #print(f"This is the additional data  we add to the tickers for the model {additional_data}")
    #print(f"VALIDATE - This is the number of training days of the train dataset {days}")

    # Get features
    matrix_features_sector = preprocessing(ticker_list_for_models, additional_data, days)

    #### Run models for all tickers selected in input  and predict

    predictions = pd.DataFrame()  # to store predictions

    for ticker in ticker_list_for_models:
        X_train, y_train, X_test, y_test, df_filtered = create_train_test_set(ticker, matrix_features_sector, lags,additional_data,days,nb_predict_days)
        MODEL.fit(X_train, y_train)
        # save the model to disk
        filename = ticker + '_model.sav'
        joblib.dump(MODEL, filename)
        temp_pred = model_predict(MODEL, ticker, X_test, y_test)

        predictions = predictions.append(temp_pred, ignore_index=True)  # this is to store in the master pandas list

    # add binary buy=1 and sell=0
    df_lagged = add_buy_sell_to_prediction(predictions)
    df_lagged  # lag_lagged is a DF containing predictions + buy and Sell label

    ticker = "*all*"
    accuracy = balanced_accuracy(ticker, df_lagged)
    print(f"Accuracy score for {ticker} is {accuracy}.")

    accuracy = []
    for ticker in ticker_list_for_models:
        accuracy.append([balanced_accuracy(ticker, df_lagged), ticker])
    DF_accuracy = pd.DataFrame(accuracy, columns=["Blc accuracy", "Ticker"])
    DF_accuracy



#functions to deal with creating and extracting features


def get_yf_dataframe(data, nbdays):
    yesterday = datetime.now() - timedelta(1)  # we want data up to yesterday
    start_date = yesterday - timedelta(days=nbdays)  # we run the model using data for nbdays
    df_res = pd.DataFrame()
    for ticker in data:
        df_tmp = si.get_data(ticker, start_date, yesterday)
        df_res[ticker] = df_tmp['close']
    return df_res


def preprocessing(ticker_list_for_models, additional_data, days):
    # this function return a matrix of features augmented with fix data for number of days

    tickers_in_sector_extended = np.concatenate((ticker_list_for_models, additional_data), axis=None)
    tickers_in_sector_extended = tickers_in_sector_extended.tolist()
    matrix_features_sector = get_yf_dataframe(tickers_in_sector_extended, days)
    # import pandas as pd; import numpy as np
    # matrix_features_sector = matrix_features_sector.reset_index()
    matrix_features_sector = matrix_features_sector.reindex(sorted(matrix_features_sector.columns), axis=1)

    ###################################
    ##### saving and reading features - can help increase processing time
    ##### to evaluate on future version
    ####################################
    # matrix_features_sector.to_csv("matrix_features_sector.csv")
    # matrix_features_sector=pd.read_csv("matrix_features_sector.csv")

    return matrix_features_sector


def create_train_test_set(ticker, features, lags,additional_data,days,nb_predict_days):
    # creating train and test set
    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        days,
        lags,
        nb_predict_days)

    df = features[[ticker] + additional_data]
    df_lagged = df.copy()
    for window in range(1, lags + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df.columns]

        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    df_lagged = df_lagged.reindex(sorted(df_lagged.columns), axis=1)

    # print(df_lagged)
    # df_lagged[ticker+"_2labels"]=np.floor(df_lagged[ticker]/df_lagged[ticker+"_lag1"]).astype(int)

    # train_set
    df_filtered = df_lagged.loc[:train_date_last]
    # X_train=df_filtered.drop(columns=[ticker, ticker+"_2labels"])
    X_train = df_filtered.drop(columns=[ticker])
    # y_train=df_filtered[ticker+"_2labels"]
    y_train = df_filtered[ticker]

    # test set
    df_filtered = df_lagged.loc[test_date_start:test_date_last]
    # X_test=df_filtered.drop(columns=[ticker, ticker+"_2labels"])
    X_test = df_filtered.drop(columns=[ticker])
    # y_test=df_filtered[ticker+"_2labels"]
    y_test = df_filtered[ticker]

    # we convert to numpy array
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, y_train, X_test, y_test, df_filtered


def initialize_data(days, lags, nb_predict_days):

    from datetime import datetime, timedelta



    # define main date in models for defining training and test sets dates.
    #yesterday = datetime.now() - timedelta(1)
    yesterday=datetime.now()
    start_date = yesterday - timedelta(days=days)
    train_date_start = start_date.strftime("%Y-%m-%d")
    train_date_last = yesterday - timedelta(days=nb_predict_days + 1)  # nombre de jours a predire
    train_date_last = train_date_last.strftime("%Y-%m-%d")

    test_date_start = yesterday - timedelta(days=nb_predict_days)
    test_date_start = test_date_start.strftime("%Y-%m-%d")
    test_date_last = yesterday.strftime("%Y-%m-%d")

    return yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last,days




# Input Output functions to save and read files

def write_list(a_list, filename_json):
    print("Started writing list data into a json file")
    with open(filename_json, "w") as fp:
        json.dump(a_list, fp)
        print("Done writing data into .json file")


# Read list to memory
def read_list(filename_json):
    # for reading also binary mode is important
    with open(filename_json, 'rb') as fp:
        n_list = json.load(fp)
        return n_list


def read_config_file():
    # return element that is in configfile. exemple additional data

    additional_data = ["^tnx", "^GSPC", "CL=F"]
    # ^TNX reasury yield is the annual return investors can expect from holding a U.S. government security with a given
    # ^GSPC tracks the performance of the stocks of 500 large-cap companies in the US"
    # CL=F crude oil pricesi.get_data(result[0][0])

    return additional_data


def create_predict_set(ticker ,features ,lags, nb_predict_days, additional_data):

    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        lags * 2, lags, nb_predict_days)
    #  List of features X_train, y_train, X_test,y_test
    df =features[[ticker ] +additional_data]
    df_lagged =df.copy()
    for window in range(1, lags + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df.columns]

        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    df_lagged =df_lagged.reindex(sorted(df_lagged.columns), axis=1)

    # training set
    df_filtered = df_lagged.loc[test_date_start:test_date_last]
    X_test =df_filtered.drop(columns=[ticker])
    y_test =df_filtered[ticker]

    # we convert to numpy array
    X_test =X_test.to_numpy()
    y_test= y_test.to_numpy()
    return X_test ,y_test ,df_filtered



#functions to predict model and calculate accuracy

def balanced_accuracy(ticker, predict):
    # put *all* to have global accuracy score for all the predictions
    # or put a ticker name
    if ticker == '*all*':
        return balanced_accuracy_score(predict['y_testb'], predict['y_predb'])
    else:
        return balanced_accuracy_score(predict['y_testb'][predict['ticker'] == ticker],
                                       predict['y_predb'][predict['ticker'] == ticker])


def model_predict(MODEL, ticker, X, y):
    y_pred = MODEL.predict(X)
    temp_pred = predictions_compile(y, y_pred, ticker)  # this is to store temporary ytest, ypredict, ticker
    return temp_pred


def predictions_compile(y_test, y_pred, ticker):
    # this allow to create a dataframe of y_test, y_predict for a given ticker
    predict_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    predict_df['ticker'] = ticker
    return predict_df
    # we need to have an initial empty dataframe to store the predictions


def add_buy_sell_to_prediction(predictions):
    # this section is to calculate the label=buy or sell
    # it is adding colum y_testb and y_predictb to data frame predictions
    # buy=1
    # sell=0

    df = predictions
    df_lagged = df.copy()
    for window in range(1, 1 + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df.columns]

        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    df_lagged['y_testb'] = np.floor(df_lagged['y_test'] / df_lagged['y_test_lag1']).astype(int)
    df_lagged['y_predb'] = np.floor(df_lagged['y_pred'] / df_lagged['y_pred_lag1']).astype(int)
    
    #*** using map function to decide buy or sell
    category={1:"Buy",0:"Sell"}
    df_lagged['y_recommend'] =df_lagged['y_predb'].map(category)
    return df_lagged



def predict_ticker(ticker_list_for_models):
    
    
    ticker_list_for_models=ticker_list_for_models.split('-')
    #note when we use flask we will use /ticker/aapl-nflx-cdw
    #so we need to split




    #print(f"Example long_test: {long_test}")
    #print(f"Example short_test: {short_test}")
    #print(f"Example not_working: {not_working}")
    #ticker_list_for_models=input_ticker


    #initialisation model
    # how many years for the model
    years = 6
    lags=30  #how many days of lags we need of this model, this is like an hyperparameter for us
    additional_data = read_config_file()  # other data needed

    nb_predict_days=30   #size of test data in number of days

    

    # if we don't have model for a ticker in list, retrain model and save
    retrain_model = False
    for ticker in ticker_list_for_models:
        if not (os.path.exists(ticker + "_model.sav")):
            retrain_model = True
            break  # this allow to continue and not go through the list if file not exist

    if retrain_model == True:
        ## Retrain all for the select list of tickers
        print(f"Training model for at least one ticker in {ticker_list_for_models}")
        Model_Train_Save(ticker_list_for_models,years,lags,additional_data,nb_predict_days)
    else:
        print(f"no model Training is needed for  {ticker_list_for_models}")
        ## get variable for start the prediction
        ##just in case, we fetch 2 time the lags so that we don't have issue when lagging


    yesterday, start_date,train_date_start,train_date_last,test_date_start,test_date_last, days=initialize_data(lags*2,lags,nb_predict_days)
    

    ### validation if we have everything
    print(f"Input ->years {years}")
    print(f"Input ->ticker list {ticker_list_for_models}")
    print(f"Input->lags {lags}")
    print(f"Input ->predict days {nb_predict_days}")
    print(f"Period->yesterday {yesterday}")
    print(f"period->train_date_start {train_date_start}")
    print(f"Period->train_date_last {train_date_last}")
    print(f"Period->test_date_start {test_date_start}")
    print(f"Period->test_date_last {test_date_last}")

    #print(f"This is the tickers for our model {ticker_list_for_models}")
    #print(f"This is the additional data  we add to the tickers for the model {additional_data}")

    df_tomorrow=preprocessing(ticker_list_for_models,additional_data,lags*2)  #add additional features
    df_tomorrow.shape

    # to store predictions
    Predictions = pd.DataFrame()

    for ticker in ticker_list_for_models:
        # load the saved model for the ticker
        filename = ticker + "_model.sav"
        loaded_model = joblib.load(filename)

        X_test, y_test, df_filtered = create_predict_set(ticker, df_tomorrow, lags, nb_predict_days,additional_data)  # this is X and
        temp_pred = model_predict(loaded_model, ticker, X_test, y_test)

        
         #===
        temp_pred.drop(labels=[0], inplace=True)


        import datetime
        from datetime import timedelta

        df_filtered2=pd.DataFrame(df_filtered[ticker])

        df_filtered2 = df_filtered2.reset_index()

    
        # Deleted 1 row in df_filtered2
        df_filtered2.drop(labels=[0], inplace=True)
      

        # Renamed columns Date
        df_filtered2.rename(columns={'index': 'Date'}, inplace=True)

        df_filtered2['Predicted for']=df_filtered2['Date']+timedelta(days=1)
        #df_filtered2['Prediction for']=df_filtered2['Date']
        df_filtered2=df_filtered2[['Date','Predicted for']]

        # Step: Copy a dataframe column


        temp_pred2=pd.concat([temp_pred,df_filtered2],axis=1)
     
    #===
        
        Predictions = Predictions.append(temp_pred2, ignore_index=True)  # this is to store in the master pandas list

        #print(f"temp_pred: \n{temp_pred}")
        #print(f"temp_pred2: \n{temp_pred2}")
        #print(f"temp_pred2: \n{temp_pred2}")
        #print(f"df_filtered2: \n{df_filtered2}")
    
        #print(f"Predictions:\n {Predictions}")
        
        
    #adding binary buy or sell to predictions dataframe
    Predictions=add_buy_sell_to_prediction(Predictions)


    ticker="*all*"
    accuracy=balanced_accuracy(ticker,Predictions)
    print(f"Accuracy score for {ticker} is {accuracy}.")



    #provide a data frame of the accuracies
    
    DF_Recommendations=[]
    for ticker in ticker_list_for_models:
        DF_Recommendations.append([ticker,"","","",balanced_accuracy(ticker,Predictions)])
        Recommendations=pd.DataFrame(DF_Recommendations, columns=["Ticker",'Predicted for','Predicted',"Recommended","Accuracy"])
    
     
    #********************************************************
    # suggestion - DF_accuracy change to DF_accuracy_recommendation 
    # suggestion - resultat change as follow: Date, Observed Value, Date Prediction, Predicted Value,Recommendation
    # suggestion - now date observed value should be change to NA
    #********************************************************
 
    
    for ticker in ticker_list_for_models:
        #results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb','y_recommend']][Predictions['ticker'] == ticker]
        
      
        results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb','y_recommend']][Predictions['ticker'] == ticker]
        #date_predict=yesterday + timedelta(1)
        date_predict=yesterday
        date_predict=date_predict.strftime("%Y/%m/%d")
        ticker_predicted=results.iloc[-1]['y_pred']  #this is the last row containing result
        ticker_recommend=results.iloc[-1]['y_recommend']  #this is the last row containing result
        Recommendations.loc[Recommendations['Ticker']==ticker,'Predicted for']=date_predict #to change content of a cell
        Recommendations.loc[Recommendations['Ticker']==ticker,'Predicted']=ticker_predicted
        Recommendations.loc[Recommendations['Ticker']==ticker,'Recommended']=ticker_recommend
        
        # print(f"Prediction for {yesterday+timedelta(1)} -- ticker: {ticker} {'**resultat**'}\n {resultat.tail()}\n\n")
    
    Results=Predictions[['ticker','Date','y_test','Predicted for','y_pred','y_recommend']]
    Results=Results.rename(columns={'ticker':'Ticker','y_test':'Observed','y_pred':'Predicted','y_recommend':'Recommended'})
    
    return Results, Recommendations



###########################################

from flask import Flask
#import git
app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return f'Welcome to VBTT v.0 ===> Please  use a better route:!\nEX: /ticker/<ticker>\n'


@app.route('/ticker/<ticker>', methods=['GET'])
def ticker(ticker):
    ticker = ticker.upper()
    filename_json="SP500.json"
    file_exists = os.path.exists("SP500.json")
    if file_exists:
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
        validation=all([([x] in SP500_list[:,:1]) for x in ticker.split("-")])
        # all allow to check if a list o bolean is tru or false . all([true, false, true....etc])
	    # X in SP50_list, etc.... .... will check if x is in my SP500 list. here all row and column 0
	    # for x in is selecting each at a time
    
        if validation==True:
            Results, Recommendations=predict_ticker(ticker)
        
        else:
            return f"Incorrect ticket, please fix or select another."
    else:
        return f"JSON does not exists - Generate JSON please"



    Results, Recommendations=predict_ticker(ticker)



    return f"Prediction for  -- ticker: {ticker} -->\n{Recommendations} \n"




if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
