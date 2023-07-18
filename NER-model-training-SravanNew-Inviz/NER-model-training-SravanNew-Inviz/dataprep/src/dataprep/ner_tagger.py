import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


class NerDataTagger:

    def __init__(self, attribute_stats, queries):
        self.uni_dict = {}
        #         self.config_file_loc = config['result_data_loc']
        self.attribute_stats = attribute_stats
        self.queries = queries

    def uni_count(self, data):
        qe = data[0].split()
        res = qe.index(data[1])
        if self.uni_dict.get(data[0], -1) < 0:
            self.uni_dict[data[0]] = res
        elif qe.count(data[1]) > 1 and self.uni_dict.get(data[0]) >= res:
            qe = qe[self.uni_dict.get(data[0]) + 1:]
            try:
                res = qe.index(data[1]) + self.uni_dict.get(data[0]) + 1
            except:
                logger.error(data)
        self.uni_dict[data[0]] = res
        return res

    def check_dict_value(self, data):
        for key in self.attribute_stats.keys():
            if self.attribute_stats[key].get(data, -1) > 0:
                return "B-" + str(key)
        return 'other'

    def check_process(self, idx):
        query = []
        word = []
        label = []
        for q in self.queries:
            qs = q.split()
            i = 0
            while i < len(qs):
                data = " ".join(qs[i:i + idx])
                i += 1
                word.append(data)
                ls = self.check_dict_value(data)
                if ls != "other":
                    i = i + idx - 1
                label.append(ls)
                query.append(q)
        return query, word, label

    def generate_training_data(self):
        for i in tqdm(range(1, 5)):
            query, word, label = self.check_process(i)

            #             print(label)
            #             break

            globals()['temp_' + str(i)] = pd.DataFrame()
            var = globals()['temp_' + str(i)]

            var['query'] = query
            var['word'] = word
            var['label'] = label

            if i == 1:
                continue

            var['word'] = var['word'].apply(lambda x: None if len(x.split()) <= (i - 1) else x)
            var['label'] = var['label'].apply(lambda x: None if x == "other" else x)

            var.dropna(inplace=True)
            var.reset_index(drop=True, inplace=True)
        #         print(temp_1)

        for i in tqdm(range(2, 5)):
            # var = pd.DataFrame(columns={'query', 'word', 'label'})
            var = []
            for j in globals()['temp_' + str(i)].values:
                data = j[1].split()
                at = j[2].split("-")[1]
                # var = var.append({'query': j[0], 'word': data[0], 'label': "B-" + at}, ignore_index=True)
                var.append({'query': j[0], 'word': data[0], 'label': "B-" + at})
                for k in range(i - 1):
                    # var = var.append({'query': j[0], 'word': data[k + 1], 'label': "I-" + at}, ignore_index=True)
                    var.append({'query': j[0], 'word': data[k + 1], 'label': "I-" + at})
            # var = var[['query', 'word', 'label']]
            var_1 = pd.DataFrame(var)
            globals()['temp_' + str(i) + '_'] = var_1.copy()

        for i in tqdm(range(1, 5)):
            self.uni_dict = {}
            if i == 1:
                temp_1['uni_cnt'] = temp_1[['query', 'word']].apply(self.uni_count, axis=1)
                temp_1.set_index(['query', 'uni_cnt'], inplace=True)
                continue

            var = globals()['temp_' + str(i) + '_'].copy()
            var['uni_cnt'] = var[['query', 'word']].apply(self.uni_count, axis=1)
            var.set_index(['query', 'uni_cnt'], inplace=True)
            temp_1.update(var)

        temp_1.reset_index(inplace=True)
        temp_1.drop("uni_cnt", axis=1, inplace=True)

        #         temp_1.to_csv(self.config_file_loc + "NER_train_data.csv", index=False)
        return temp_1

    def check_dict_value_v2(self, data):
        for key in self.attribute_stats.keys():
            if data in self.attribute_stats[key]:
                return key
        return 'other'

    def tag_v2(self):
        logger.info('Tagging click stream queries using ner attributes values...')
        max_win = 4
        min_win = 1

        globals()['temp_data'] = {
            'qs': [],
            'qs_words': [],
            'qs_labels': []
        }

        for query_i in tqdm(range(0, len(self.queries))):
            query = self.queries[query_i]
            words = query.split()
            labels = ['other'] * len(words)

            for win in range(min_win, max_win + 1):
                # for i in range(0, len(words) - win + 1):
                i = 0
                while i < len(words) - win + 1:

                    pending = True  # any(map(lambda x: labels[x] == 'other', range(i, i + win)))

                    if pending:
                        phrase = " ".join(words[i:i + win])
                        # phrase_labels = labels[i:i + win]
                        # pending = not any(filter(lambda x: x != 'other', phrase_labels))
                        # if pending:

                        label = self.check_dict_value_v2(phrase)
                        if label != 'other':
                            for ix in range(i, i + win):

                                if ix == i:  # and (ix == 0 or labels[ix] == 'other' or labels[ix].startswith('B-')):
                                    labels[ix] = 'B-' + label
                                else:
                                    labels[ix] = 'I-' + label

                            i += win - 1
                    i += 1

            # if not any(filter(lambda x: 'brand' in x, labels)):
            #     if words[0] == 'and':
            #         labels[0] = 'B-brand'
            #     elif words[-1] == 'and':
            #         labels[-1] = 'B-brand'
            #     elif words[0] == 'w':
            #         labels[0] = 'B-brand'

            globals()['temp_data']['qs'].extend([query] * len(words))
            globals()['temp_data']['qs_words'].extend(words)
            globals()['temp_data']['qs_labels'].extend(labels)

        df = pd.DataFrame(data={
            'query': globals()['temp_data']['qs'],
            'word': globals()['temp_data']['qs_words'],
            'label': globals()['temp_data']['qs_labels']
        })
        # df = df.explode(['word', 'label'])
        return df
