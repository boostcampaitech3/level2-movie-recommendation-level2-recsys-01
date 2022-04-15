import collections
import pandas as pd
from tqdm import tqdm

data_path = 'lightfm.csv'
df_light = pd.read_csv(data_path)
data_path = 'sasrec.csv'
df_sas = pd.read_csv(data_path)
data_path = 'recvae.csv'
df_rec = pd.read_csv(data_path)

user_ids = df_rec['user'].unique()

user_output=[]
item_output=[]
for u in tqdm(user_ids):
    list_light=list(df_light[df_light['user']==u].iloc[:,1])
    list_sas=list(df_sas[df_sas['user']==u].iloc[:,1])
    list_rec=list(df_rec[df_rec['user']==u].iloc[:,1])
    total=list_rec+list_light+list_sas
    a=collections.Counter(total)
    a=list((a.most_common(10)))
    user_output.extend([u]*10)
    for i in a:
        item_output.append(i[0])

submission = pd.read_csv("sample_submission.csv")
submission["user"] = user_output
submission["item"] = item_output
submission.to_csv("hardvoting.csv", index=False)