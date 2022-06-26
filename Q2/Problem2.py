import pandas as pd
from math import sqrt 
import numpy as np 
import os 


class Strategy: 
    def __init__(self,data):
        
        self.data = data
        self.annual_days = 365

        '''
        Your strategy has to coded in this section 
        A sample strategy of moving average crossover has been provided to you. You can uncomment the same and run the code for checking data output 
        You strategy module should return a signal dataframe
        '''
    
    def sma(self, Data, window):
        return Data.rolling(window).mean()

    def bb(self, Data, sm, window):
        std = Data.rolling(window).std()
        upper_bb = sm + std*2
        lower_bb = sm - std*2
        return upper_bb, lower_bb

    def implement_strategy(self, Data, lower_bb, upper_bb):
        bb_signal = []
        signal = 0
        for i in range(len(Data)):
            if (Data[i-1] > lower_bb[i-1]) and (Data[i] < lower_bb[i]):
                if signal != 1:
                    signal = 1
                    bb_signal.append(signal)
                else:
                    bb_signal.append(0)
            elif (Data[i-1] < upper_bb[i-1]) and (Data[i] > upper_bb[i]):
                if signal != -1:
                    signal = -1
                    bb_signal.append(signal)
                else:
                    bb_signal.append(0)
            else:
                bb_signal.append(0)
        return bb_signal


    def strategy(self):
           
        # short_window = 15
        # long_window = 30
    
        # signal = self.data.rolling(short_window).mean() - self.data.rolling(long_window).mean()
        # return signal
        # change = self.data.pct_change()
        # return change.shift(-1)'
        sma_series = pd.DataFrame()
        win = 30
        sma_series['sma_Series1'] = self.sma(self.data['Series1'], win)
        sma_series['sma_Series2'] = self.sma(self.data['Series2'], win)
        sma_series['sma_Series3'] = self.sma(self.data['Series3'], win)
        sma_series['sma_Series4'] = self.sma(self.data['Series4'], win)
        sma_series['sma_Series5'] = self.sma(self.data['Series5'], win)
        sma_series['sma_Series6'] = self.sma(self.data['Series6'], win)
        sma_series['sma_Series7'] = self.sma(self.data['Series7'], win)
        sma_series['sma_Series8'] = self.sma(self.data['Series8'], win)
        sma_series['sma_Series9'] = self.sma(self.data['Series9'], win)
        sma_series['sma_Series10'] = self.sma(self.data['Series10'], win)


        bollinger_bands = pd.DataFrame()
        bollinger_bands['upper_bb_Series1'], bollinger_bands['lower_bb_Series1'] = self.bb(self.data['Series1'], sma_series['sma_Series1'], win)
        bollinger_bands['upper_bb_Series2'], bollinger_bands['lower_bb_Series2'] = self.bb(self.data['Series2'], sma_series['sma_Series2'], win)
        bollinger_bands['upper_bb_Series3'], bollinger_bands['lower_bb_Series3'] = self.bb(self.data['Series3'], sma_series['sma_Series3'], win)
        bollinger_bands['upper_bb_Series4'], bollinger_bands['lower_bb_Series4'] = self.bb(self.data['Series4'], sma_series['sma_Series4'], win)
        bollinger_bands['upper_bb_Series5'], bollinger_bands['lower_bb_Series5'] = self.bb(self.data['Series5'], sma_series['sma_Series5'], win)   
        bollinger_bands['upper_bb_Series6'], bollinger_bands['lower_bb_Series6'] = self.bb(self.data['Series6'], sma_series['sma_Series6'], win)
        bollinger_bands['upper_bb_Series7'], bollinger_bands['lower_bb_Series7'] = self.bb(self.data['Series7'], sma_series['sma_Series7'], win)
        bollinger_bands['upper_bb_Series8'], bollinger_bands['lower_bb_Series8'] = self.bb(self.data['Series8'], sma_series['sma_Series8'], win)
        bollinger_bands['upper_bb_Series9'], bollinger_bands['lower_bb_Series9'] = self.bb(self.data['Series9'], sma_series['sma_Series9'], win)
        bollinger_bands['upper_bb_Series10'], bollinger_bands['lower_bb_Series10'] = self.bb(self.data['Series10'], sma_series['sma_Series10'], win)

        strat = self.data
        strat['Series1'] = self.implement_strategy(self.data['Series1'], bollinger_bands['lower_bb_Series1'], bollinger_bands['upper_bb_Series1'])
        strat['Series2'] = self.implement_strategy(self.data['Series2'], bollinger_bands['lower_bb_Series2'], bollinger_bands['upper_bb_Series2'])
        strat['Series3'] = self.implement_strategy(self.data['Series3'], bollinger_bands['lower_bb_Series3'], bollinger_bands['upper_bb_Series3'])
        strat['Series4'] = self.implement_strategy(self.data['Series4'], bollinger_bands['lower_bb_Series4'], bollinger_bands['upper_bb_Series4'])
        strat['Series5'] = self.implement_strategy(self.data['Series5'], bollinger_bands['lower_bb_Series5'], bollinger_bands['upper_bb_Series5'])
        strat['Series6'] = self.implement_strategy(self.data['Series6'], bollinger_bands['lower_bb_Series6'], bollinger_bands['upper_bb_Series6'])
        strat['Series7'] = self.implement_strategy(self.data['Series7'], bollinger_bands['lower_bb_Series7'], bollinger_bands['upper_bb_Series7'])
        strat['Series8'] = self.implement_strategy(self.data['Series8'], bollinger_bands['lower_bb_Series8'], bollinger_bands['upper_bb_Series8'])
        strat['Series9'] = self.implement_strategy(self.data['Series9'], bollinger_bands['lower_bb_Series9'], bollinger_bands['upper_bb_Series9'])
        strat['Series10'] = self.implement_strategy(self.data['Series10'], bollinger_bands['lower_bb_Series10'], bollinger_bands['upper_bb_Series10'])

        return strat
        '''
        This module computes the daily asset returns based on long/short position and stores them in a dataframe 
        '''
    def process(self):
        returns = self.data.pct_change()
        self.signal = self.strategy()
        self.position = self.signal.apply(np.sign)
        self.asset_returns = (self.position.shift(1)*returns)
        return self.asset_returns

        '''
        This module computes the overall portfolio returns, asset portfolio value and overall portfolio values 
        '''

    def portfolio(self):
        asset_returns = self.process()
        self.portfolio_return = asset_returns.sum(axis=1)
        self.portfolio = 100*(1+self.asset_returns.cumsum())
        self.portfolio['Portfolio'] = 100*(1+self.portfolio_return.cumsum())
        return self.portfolio

        '''
        This module computes the sharpe ratio for the strategy
        '''

    def stats(self):
        stats = pd.Series()
        self.index = self.portfolio()
        stats['Start'] = self.index.index[0]
        stats['End'] = self.index.index[-1]
        stats['Duration'] = pd.to_datetime(stats['End']) - pd.to_datetime(stats['Start'])
        annualized_return = self.portfolio_return.mean()*self.annual_days
        stats['Annualized Return'] = annualized_return
        stats['Annualized Volatility'] = self.portfolio_return.std()*sqrt(self.annual_days)
        stats['Sharpe Ratio'] = stats['Annualized Return'] / stats['Annualized Volatility']
        return stats
        
if __name__ == '__main__':

    """ 
    Function to read data from csv file 
    """
    data = pd.read_csv(os.path.join(os.getcwd(),'Data.csv'),index_col='Date')
    result = Strategy(data)
    res = result.stats()
    res.to_csv(os.path.join(os.getcwd(),'Result.csv'),header=False)



