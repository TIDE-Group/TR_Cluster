import pandas as pd
import numpy as np
from hot_topic import HotTopic
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('./cnn_labeled.xlsx')
labels = df['label'].unique()
topic_df_list = [df[df['label'] == label] for label in labels]

htt = HotTopic(topic_df_list, 'week', 'week')

print(htt.total_score_rank)
for idx, df in htt.period_score_rank.items():
    print('\n{}:'.format(idx))
    print(df)

pass