import json
import string

import numpy as np
import pandas as pd
from loguru import logger

from src.dataprep.attributes_generator import AttributesGenerator
from src.dataprep.data_combiner import DataCombiner
from src.dataprep.ner_tagger import NerDataTagger
from src.schema.schema import JobArgs
from src.storage.path_utils import PathUtils


class NERTrainDataGeneration():

    @staticmethod
    def get_common_attrs(attrs) -> dict:
        common = {}

        keys = list(attrs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):

                sim = set.intersection(set(attrs[keys[i]]), set(attrs[keys[j]]))
                if len(sim) > 0:
                    common[(keys[i], keys[j])] = sim

        return common

    @staticmethod
    def filter_common(attrs, data):
        logger.info('Filtering query that contains attribute common values...')

        common_attrs = NERTrainDataGeneration.get_common_attrs(attrs)
        values = [v for values in common_attrs.values() for v in values]
        q_to_remove = data[(data['word'].isin(values)) & (~data['label'].str.contains('other'))]['query'].unique()

        # Filter Phrases Query
        values = [v for values in common_attrs.values() for v in values if ' ' in v]
        for phrase in values:
            for token in phrase.split(' '):
                qs = data[(data['query'].str.contains(phrase)) & (data['word'].str.contains(token)) & (
                    ~data['label'].str.contains('other'))]['query'].unique()
                q_to_remove = np.concatenate([q_to_remove, qs])

        unique_queries = list(set(q_to_remove))

        logger.info(f'Total queries to filter/remove: {str(len(unique_queries))}')

        data.drop(data[data['query'].isin(unique_queries)].index, inplace=True)

        logger.info(f'After filter total queries: {str(data["query"].nunique())}')

        return data

    @staticmethod
    def add_attribute_values(attrs, df):
        logger.info('Adding ner attributes values in train dataset...')
        queries = set(df['query'])

        data = []

        index = 1
        for i in [k for k in attrs]:
            for j in attrs[i]:
                if j not in queries:
                    if not j.replace(' ', '').isdigit():
                        rec = {}
                        rec['index'] = index
                        rec['query'] = j
                        rec['word'] = j.split()
                        rec['label'] = [('B' if k == 0 else 'I') + '-' + i for k in range(len(rec['word']))]
                        data.append(rec)
                        index += 1

        attr_df = pd.DataFrame(data=data)

        attr_df = attr_df[~attr_df.duplicated(subset=['query'])]
        attr_df = attr_df[['query', 'word', 'label']].set_index(['query']).apply(pd.Series.explode).reset_index()

        punctuation = [i for i in string.punctuation if i != '&']
        invalid_queries = set(attr_df[attr_df['word'].apply(lambda x: x in punctuation)]['query'])
        attr_df = attr_df[~attr_df['query'].isin(invalid_queries)].dropna()

        df = pd.concat([df, attr_df], ignore_index=True)

        return df


    def get_queries(self):
        df = pd.read_csv(PathUtils.CLICK_STREAM_DATASET)
        df['query'] = df['query'].apply(lambda x: x.lower())
        df = df.dropna()
        return df['query'].tolist()

    def generate(self, job_args: JobArgs):

        # Generate ner attribute map
        attributes_generator = AttributesGenerator()
        facet_map, attrs = attributes_generator.generate_ner_attributes_map(job_args)

        json.dump(facet_map, open('facet_value_map.json', 'w'))
        json.dump(attrs, open('ner_attributes_map.json', 'w'))

        # Get user searched query
        queries = self.get_queries()

        # Tag searched query using ner attribute value map
        ner_tagger = NerDataTagger(attrs, queries)
        df = ner_tagger.tag_v2()

        # Remove common val
        df = NERTrainDataGeneration.filter_common(attrs, df)
        df = NERTrainDataGeneration.add_attribute_values(attrs, df)

        df.to_csv("ner_train_data_cs.csv", index=False)

        df = DataCombiner().combine_cs_gs_dataset(ner_train_df=df)

        df.to_csv("ner_train_data_cs_gs.csv", index=False)




