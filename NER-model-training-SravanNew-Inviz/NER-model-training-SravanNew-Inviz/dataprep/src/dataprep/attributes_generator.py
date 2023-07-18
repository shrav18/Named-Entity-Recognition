import itertools
import json
import re
import string
from itertools import islice
from typing import Dict

import nltk
import pandas as pd
from loguru import logger
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from tqdm.auto import tqdm
tqdm.pandas()

from src.storage.path_utils import PathUtils

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
wnl = WordNetLemmatizer()

import inflect

from src.schema.schema import JobArgs, AttributeConfig


# Clearn Value Functions
REPLACE_GROUPING_BRACKETS = re.compile(r"[\\([{})\]]")
SPLIT_BY_VALUE_SEPARATOR = re.compile(r"[\\|:,+]")

# AttributesVariationGenerator
inflect_obj = inflect.engine()
p = re.compile(r'[' + string.punctuation + ']')
and_p = re.compile(r'( and | n |\s*&\s*)+')
spaces = re.compile(r'\s+')
start_word = ['a', 'an', 'the', 'n', 'and', '&', 'by', 'of', 'from', '-', 'smart']
color_start_words = ['dark', 'light', 'off', 'neutral', 'natuaral', 'warm'] + start_word
filter_words = {'by', 'of', 'the', 'from', 'combo'}

class AttributesVariationGenerator:


    def __init__(self, attr_configs: Dict[str, AttributeConfig]) -> None:
        self.attr_configs = attr_configs

    def get_invalid_values(self, attr):
        if attr in self.attr_configs:
            return self.attr_configs.get(attr).invalid_values or []
        return []

    def generate_singular_plural(self, text: str, attr, singular=True) -> str:
        temp = []
        for j in text.split():
            if len(j) > 2 and j not in self.get_invalid_values(attr) and j not in start_word and j not in filter_words:
                a = inflect_obj.singular_noun(j) if singular else inflect_obj.plural_noun(j)
            else:
                a = False
            if not singular and a and (a.endswith('ss') or a.endswith('se') or j.endswith('es')):
                a = False

            if a is not False:
                temp.append(a)
            else:
                temp.append(j)

        #     print(text, "  =  " ," ".join(temp))

        return " ".join(temp)

    def generate_variations(self, v, out=None, attr='brand'):
        if out is None:
            out = {v.lower()}
        elif v not in out:
            out.add(v)
        else:
            return out

        tokens = v.split()
        # out.add(''.join([i.strip() for i in p.split(v)]))
        t = ''.join([i.strip() for i in p.split(v)])
        if t != v and t not in out:
            self.generate_variations(t, out, attr)

        # out.add(''.join([(i.strip() + ' ' if len(i.strip()) != 1 else i.strip()) for i in p.split(v)]).strip())
        t = ''.join([(i.strip() + ' ' if len(i.strip()) != 1 else i.strip()) for i in p.split(v)]).strip()
        if t != v and t not in out:
            self.generate_variations(t, out, attr)

        # out.add(p.sub('', v))
        t = p.sub('', v)
        if t != v and t not in out:
            self.generate_variations(t, out, attr)

        for i in [' ', ' and ', ' & ', '&', ' n ']:
            if i in v:
                t = and_p.sub(i, v)
                if t not in out:
                    self.generate_variations(t, out, attr)

        vs = and_p.split(v)
        if len(vs) >= 3:
            if attr in ['color', 'style', 'product_type']:
                self.generate_variations(vs[0], out, attr)
                self.generate_variations(vs[2], out, attr)
                if ' ' not in vs[2]:
                    self.generate_variations(' '.join([i.strip() for i in reversed(vs)]), out, attr)

            if attr in ['color', 'style', 'product_type']:
                for ind, j in enumerate(and_p.split(v)):
                    j = j.strip()
                    if ind % 2 == 0 and j not in out:
                        self.generate_variations(j, out, attr)

            for i in [' ', ' and ', ' & ', '&', ' n ']:
                self.generate_variations(vs[0] + i + vs[2], out, attr)

        #     if ' ' in v and attr in ['brand', 'product_type']:
        #         # self.generate_variations(spaces.sub('', v), out, attr)
        #         for win in range(2, 4):
        #             for i in range(len(tokens)-(win-1)):
        #                 if len(set(tokens[i:i+win]).intersection(start_word)) == 0:
        #                     t = ' '.join(tokens[:i] + [''.join(tokens[i:i+win])] + tokens[i+win:])
        #                     if t not in out:
        #                         self.generate_variations(t, out, attr)

        if "'s" in v:
            self.generate_variations(v.replace("'s", ''), out, attr)
            self.generate_variations(v.replace("'s", 's'), out, attr)

        if "by" in tokens[1:-1]:
            ni = v.split(" by ")
            self.generate_variations(ni[0], out, attr)
            self.generate_variations(ni[1], out, attr)
            self.generate_variations(ni[0] + ' ' + ni[1], out, attr)

        if "with" in tokens[1:-1]:
            ni = v.split(" with ")
            self.generate_variations(ni[0], out, attr)

        sw = start_word if attr != 'color' else color_start_words
        if len(tokens) > 1 and tokens[0] in sw and tokens[1] not in sw:
            ni = ' '.join(tokens[1:])
            self.generate_variations(ni, out, attr)

        if len(tokens) > 1:
            ni = ' '.join([i for i in tokens if i not in filter_words])
            if ni != v and len(ni) >= 5:
                self.generate_variations(ni, out, attr)

        # create variations for singular form
        words_array = [list({i, wnl.lemmatize(i), self.generate_singular_plural(i, attr)}) for i in tokens]
        for words in itertools.product(*words_array):
            n_val = ' '.join(words)
            if n_val not in out:
                self.generate_variations(n_val, out, attr)

        words_array = [
            list({i, wnl.lemmatize(i), self.generate_singular_plural(i, attr), self.generate_singular_plural(i, attr, singular=False)}) for i
            in tokens]
        for words in itertools.product(*words_array):
            n_val = ' '.join(words)
            if n_val not in out:
                out.add(n_val)

        return out  # {i.strip() for i in out}

    def generate_variations_(self, val, attr:str):
        try:
            out = self.generate_variations(val.lower(), attr=attr)
            return list({' '.join(i.split()) for i in out if i not in self.get_invalid_values(attr)})
        except Exception as e:
            print(val)
            raise e

    # def generate_list_variations(self, vals, variations, attr):
    #     fmap = dict() if variations is None else variations
    #     for i in vals:
    #         vs = self.generate_variations_(i, attr)
    #         for j in vs:
    #             if j not in fmap:
    #                 fmap[j] = []
    #             if i not in fmap[j]:
    #                 fmap[j].append(i)
    #
    #     if attr == 'product_type':
    #         frmap = dict()
    #         for i, j in fmap.items():
    #             for k in j:
    #                 if k not in frmap:
    #                     frmap[k] = []
    #                 if i not in frmap[k]:
    #                     frmap[k].append(i)
    #
    #         for i, vs in frmap.items():
    #             cvs = {k for j in vs for k in fmap.get(j)}
    #             for j in vs:
    #                 for k in cvs:
    #                     if k not in fmap[j]:
    #                         fmap[j].append(k)
    #
    #     return fmap

class AttributesGenerator:

    @staticmethod
    def catalog_file_reader(file_path):
        with open(file_path) as file:

            while True:

                next_n_lines = list(islice(file, 10_000))

                if not next_n_lines:
                    break
                else:
                    for line in next_n_lines:
                        try:
                            if len(line) == 0:
                                yield {}
                            else:
                                yield json.loads(line)
                        except Exception as e:
                            logger.error('failed to parse catalog line: ' + line)
                            raise e

    @staticmethod
    def _clean_value(val: str, attr_conf: AttributeConfig):
        val = unidecode(val.lower())
        out = []

        st_words = attr_conf.invalid_values or []
        part_st_words = attr_conf.invalid_partial_values or []

        for i in SPLIT_BY_VALUE_SEPARATOR.split(val):
            i = i.strip()
            if i not in st_words and not any(filter(lambda x: x in i, part_st_words)):
                i = REPLACE_GROUPING_BRACKETS.sub("", i)
                if i not in st_words and not any(filter(lambda x: x in i, part_st_words)):
                    out.append(i)

        sp = val.split(':')
        if len(sp) == 2 and len(sp[1]) > 2:
            out.extend(AttributesGenerator._clean_value(sp[1].strip(), attr_conf))

        return list(
            {i for i in out if len(i) < 50 and i not in st_words})  # if len(out) > 0 or val in st_words else [val]

    @staticmethod
    def clean_value(val, attr_conf: AttributeConfig):
        val = AttributesGenerator._clean_value(val, attr_conf)
        return val if len(val) > 0 else None


    @staticmethod
    def generate_ner_attributes_map(job_args: JobArgs):
        logger.info("Preparing NER attribute value map")

        # map source fields from catalog dataset
        data = []
        for rec in tqdm(AttributesGenerator.catalog_file_reader(PathUtils.CATALOG_DATASET)):

            out = {'brand': {rec.get('brand')}}
            for i in job_args.field_map:
                kv = rec['attributes'] if i not in rec else rec
                if i in kv and kv[i] is not None:
                    if job_args.field_map[i] in out:
                        out[job_args.field_map[i]].update(set(kv[i]) if type(kv[i]) == list else {kv[i]})
                    else:
                        out[job_args.field_map[i]] = set(kv[i]) if type(kv[i]) == list else {kv[i]}

            data.append(out)

        # df = pd.DataFrame(data=data)
        dataset = []
        for rec in data:
            for attr in rec:
                dataset.append({
                    'attribute': attr,
                    'value': rec[attr]
                })

        for name in job_args.attr_configs:
            if job_args.attr_configs[name].fixed_values:
                dataset.append({
                    'attribute': name,
                    'value': job_args.attr_configs[name].fixed_values
                })

        df = pd.DataFrame(data=dataset)
        df = df.explode('value')
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)


        # Clean value
        logger.info("Cleaning attribute values")
        df['clean_values'] = df.apply(lambda x: AttributesGenerator.clean_value(str(x['value']), job_args.attr_configs.get(x['attribute'])), axis=1)
        df.dropna(inplace=True)

        # Generate value variations
        logger.info("Generating value variations")
        variation_generator = AttributesVariationGenerator(job_args.attr_configs)
        df['value_variations'] = df.progress_apply(lambda x: list({j for i in x['clean_values'] for j in variation_generator.generate_variations_(i, x['attribute'])}), axis=1)
        df = df[['attribute','value', 'value_variations']]
        df = df.explode('value_variations')
        df.dropna(inplace=True)
        df = df.groupby(['attribute', 'value_variations']).agg({'value': list})

        # prepare facet value map
        logger.info("Preparing facet value map")
        facet_map = dict()
        for i, rec in df.reset_index().iterrows():
            att = rec['attribute']
            value = rec['value']
            value_variations = rec['value_variations']
            if att not in facet_map:
                facet_map[att] = dict()
            facet_map[att][value_variations] = value

        # prepare ner attribute map
        logger.info("Preparing ner attributes value map")
        ner_attrs = {}
        for att in facet_map:
            ner_attrs[att] = list(facet_map[att])

        return facet_map, ner_attrs