from typing import List, Union
import pandas as pd
import torch
import numpy as np

def calc_period_id_df(df:pd.DataFrame, granularity:str) :
    period_slice = {'week': 2, 'day': 3}
    if granularity in ['week', 'day']:
        df['period'] = df['date_calendar'].apply(eval)
        df['period'] = df['period'].apply(lambda x : x[:period_slice[granularity]])
    elif granularity in ['month']:
        df['period'] = df['date'].apply(lambda x : (x.year, x.month))
    return df

def calc_period_id_s(s:pd.Series, granularity:str):
    period_slice = {'week': 2, 'day': 3}
    if granularity in ['week', 'day']:
        period = s.apply(lambda x : x.isocalendar()).apply(lambda x : x[:period_slice[granularity]])
    elif granularity in ['month']:
        period = s.apply(lambda x : x.isocalendar()).apply(lambda x : (x.year, x.month))
    return period

class HotTopic(object):
    def __init__(self, topic_df_list:List[pd.DataFrame], granularity:str, result_granularity:str):
        """
        topic_df_list: 每一个DF为同一个话题
        """
        super().__init__()
        assert granularity in ['week', 'day', 'month'], "计算的粒度选择错误"
        assert result_granularity in ['week', 'day', 'month'], "展示结果的粒度选择错误"
        self.granularity = granularity
        self.result_granularity = result_granularity
        self.granularity_span = {'week':7, 'day':1, 'month':30}[self.granularity]
        self.result_granularity_span = {'week':7, 'day':1, 'month':30}[self.result_granularity]
                
        self.topic_df_list = [calc_period_id_df(df, self.granularity) for df in topic_df_list]
        self.feature = 2 # sf + rd
        self.features = ['sf', 'rd']
        self.score_list = [self.__calc_score__(topic_df, idx) for  idx, topic_df in enumerate(self.topic_df_list)]
        self.score_df = pd.concat(self.score_list, ignore_index=True, axis=0)
        self.total_score_rank = self._calc_hot_score_(self.score_df)
        self.period_score_rank = self._calc_hot_score_period_(self.score_df)

        
    def __calc_score__(self, topic_df:pd.DataFrame, topic_id:int):
        # granularity_set = eval("topic_df['date'].dt.%s.unique()"%self.granularity) #时间跨度
        granularity_set = topic_df['period'].unique()
        # score = np.zeros((len(granularity_set), self.feature)) #[时间跨度, 2]
        score = pd.DataFrame(columns=['topic_id', 'period_id', 'result_period_id'] + self.features)

        for i in range(len(granularity_set)): #分别计算每一个时间跨度的特征值
            period_id = granularity_set[i]
            # period_topic_df = eval("topic_df[topic_df['date'].dt.%s == time_span_id]"%self.granularity)
            period_topic_df = topic_df[topic_df['period'] == period_id]
            period_topic_df['day'] = calc_period_id_s(period_topic_df['date'], 'day')
            period_topic_df['result_period_id'] = calc_period_id_s(period_topic_df['date'], self.result_granularity)

            #某个话题中属于同一时间段下的所有文章
            sf = len(period_topic_df)
            rd = len(period_topic_df['day'].unique())
            result_period_id = period_topic_df['result_period_id'].value_counts().idxmax()
            
            score = score.append(
                pd.Series({
                    'topic_id': topic_id, 'period_id':period_id, 
                    'result_period_id':result_period_id, 'sf':sf, 'rd':rd
                }),
                ignore_index=True
            )
        return score

    def _calc_hot_score_(self, score_df:pd.DataFrame):
        period_ids = score_df['period_id'].unique()
        topic_ids = score_df['topic_id'].unique()

        D = {idx : score_df[score_df['period_id'] == idx]['sf'].sum() for idx in period_ids}

        def __add_D_column__(x):
            period_id = x['period_id']
            return D[period_id]
        score_df['D'] = score_df.apply(__add_D_column__, axis=1)

        def __calc_single_period_score__(x):
            return 10 * x['sf'] / x['D'] + x['rd'] / self.granularity_span
        
        score_df['score'] = score_df.apply(__calc_single_period_score__, axis=1)
        # total_score = {idx:score_df[score_df['topic_id'] == idx]['score'].sum() for idx in topic_ids}
        total_score = {
            'topic_id':[idx for idx in topic_ids],
            'hot_topic_score':[score_df[score_df['topic_id'] == idx]['score'].sum() for idx in topic_ids]
        }

        return pd.DataFrame(total_score).sort_values(ascending=False, by=['hot_topic_score'])

    def _calc_hot_score_period_(self, score_df:pd.DataFrame):
        result_period_ids = score_df['result_period_id'].unique()
        period_score_df_dict = {idx:score_df[score_df['result_period_id'] == idx] for idx in result_period_ids}
        period_score_rank = {idx:self._calc_hot_score_(df) for idx, df in period_score_df_dict.items()}
        return period_score_rank
        




    
    

            



        
