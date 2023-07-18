import ast
import glob
from src.core.config import  APP_ID, MODEL_ID
from src.storage.path_utils import PathUtils
import json
import pandas as pd
from loguru import logger

class DataCombiner:

    def get_att(x,field_maps):
            temp_dict = {}
            for i in x:
                if i in field_maps.keys():
                    temp_dict[field_maps[i]] = x[i][0].lower()
            return temp_dict

    def dict_without_keys(d,invalid_key):
        return {x: d[x] for x in d if x not in invalid_key}


    def read_generated_suggestion_datasets(self):

        df = pd.read_json(PathUtils.GENERATED_SUGGESTION_DATASET, lines=True)
        df = df[['search_term','filters']]
        df = df[df['filters'].apply(lambda x: len(x)) != 0]
        field_maps = json.load(open('fieldmap.json','r'))
        df['att'] = df['filters'].apply(lambda x :DataCombiner.get_att(x,field_maps))
        df = df[['search_term','att']]
        df.columns = ['query','attributes']
        df.to_csv('generated_suggestions.csv', index=False)
        
        all_files = glob.glob("generated_suggestion*.csv")

        df_from_each_file = [pd.read_csv(f) for f in all_files]
        if len(df_from_each_file) > 0:
            concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
            concatenated_df = concatenated_df.drop_duplicates(subset=["query"]).reset_index(drop=True)

            logger.info(f"Total Generated suggestions: {str(concatenated_df.shape[0])}")

            if concatenated_df.shape[0] > 10_00_00_000:  # 50_00_000: 17_68_874
                logger.info("Removing data from Generated suggestions")

                concatenated_df['sorted_query'] = concatenated_df['query'].apply(lambda q: ' '.join(sorted(q.split(' '))))
                # df1 = concatenated_df.drop_duplicates(subset=['sorted_query'], keep='first')
                # df2 = concatenated_df.drop_duplicates(subset=['sorted_query'], keep='last')
                #
                # concatenated_df = pd.concat([df1, df2]).drop_duplicates(subset=['sorted_query', 'query'],
                #                                                         keep='first').reset_index()

                concatenated_df['counter'] = concatenated_df.groupby(['sorted_query']).cumcount()
                concatenated_df = concatenated_df[concatenated_df['counter'] % 2 == 0]

                logger.info(f"Now Total Generated suggestions: {str(concatenated_df.shape[0])}")

            return concatenated_df[['query', 'attributes']].dropna()
        return None

    @staticmethod
    def find_next_slice(seq, subseq, tags):
        n = len(seq)
        m = len(subseq)
        for i in range(n - m + 1):
            if seq[i:i + m] == subseq:
                u_tags = list(set(tags[i:i + m]))
                if len(u_tags) == 1 and u_tags[0] == 'other':
                    return i
        return -1

    @staticmethod
    def map_attributes_to_word_label(row):
        att_dict = ast.literal_eval(row['attributes'])
        query = row['query']
        tokens = query.split(" ")
        tags = ['other'] * len(tokens)

        try:
            sorted_attrs = sorted(att_dict, key=lambda k: len(str(att_dict[k])), reverse=True)

            for att in sorted_attrs:

                if att != 'pattern_score':

                    q_tokens = att_dict[att].split(" ")

                    index = DataCombiner.find_next_slice(tokens, q_tokens, tags)

                    if index != -1:

                        tags[index] = 'B-' + att

                        for i in range(index + 1, index + len(q_tokens)):
                            tags[i] = 'I-' + att
        #                 else:
        #                     print(query)
        #                     print(row['attributes'])

        except Exception as e:
            print(query)
            print(row['attributes'])
            raise e

        return pd.Series([query, tokens, tags], index=['query', 'word', 'label'])

    def combine_cs_gs_dataset(self, ner_train_df=None):

        if ner_train_df is None:
            ner_train_df = pd.read_csv("ner_train_data_cs.csv").dropna()

        ner_train_df['source'] = "cs"

        gs_df = self.read_generated_suggestion_datasets()
        if gs_df is not None:

            nunique = ner_train_df['query'].nunique()
            logger.info("Ner Train dataset size: " + str(nunique))

            cond = ner_train_df['query'].isin(gs_df['query'])
            ner_train_df.drop(cond[cond].index, inplace=True)

            gs_df = gs_df.progress_apply(DataCombiner.map_attributes_to_word_label, axis=1)
            gs_df = gs_df.explode(['word', 'label']).dropna()
            gs_df['source'] = "gs"

            if ner_train_df.shape[0] + gs_df.shape[0] < 50_00_000:
                logger.info(f"Multiplying Generated suggestions by 2...")
                new_df_1 = gs_df.copy()
                new_df_2 = gs_df.copy()

                new_df_1['query'] = new_df_1['query'].apply(lambda q: '1 ' + q)
                new_df_2['query'] = new_df_2['query'].apply(lambda q: '2 ' + q)

                # concat old dataset and new dataset
                concatenated_df = pd.concat([ner_train_df, new_df_1, new_df_2], ignore_index=True)
            else:
                concatenated_df = pd.concat([ner_train_df, gs_df], ignore_index=True)

            concatenated_df.dropna(inplace=True, subset=['query', 'word', 'label', 'source'])

            nunique = concatenated_df['query'].nunique()
            logger.info("Combined Dataset Size: " + str(nunique))

            return concatenated_df
        return ner_train_df