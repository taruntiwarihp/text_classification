from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SentimentSentimentDataset(Dataset):

    class_to_id = {
        'positive': 0,
        'negative': 1
    }

    def __init__(self, root, tokenizer = None, max_len=512):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data, self.label = self.process_data(root)

    def process_data(self, root):

        df_temp = pd.read_csv(root)
        df = df_temp[['review', 'sentiment']]
        df = df.astype(str)
        df.dropna(inplace=True)
        df = df.astype(str)
        df = df[df['sentiment'].isin(list(self.class_to_id.keys()))]

        df['sentiment'] = df.sentiment.map(self.class_to_id)
        df_final = df.copy()
        df_final = df_final.reindex(np.random.permutation(df_final.index))

        return df_final['review'].values, df_final['sentiment'].values


    def __getitem__(self, idx):
        description = str(self.data[idx])
        targets = int(self.label[idx])

        data = self.tokenizer.encode_plus(
            description,
            max_length=self.max_len,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        return {
            'ids': data['input_ids'], #.squeeze(),
            'mask': data['attention_mask'], #.squeeze(),
            'token_type_ids': data['token_type_ids'], #.squeeze(),
            'labels': targets,
            # 'len': torch.tensor(len(self.label), dtype=torch.long)
        }

    def __len__(self):
        return len(self.label)