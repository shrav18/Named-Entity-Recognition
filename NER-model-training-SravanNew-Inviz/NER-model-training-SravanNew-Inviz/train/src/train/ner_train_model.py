import math
import os.path

import torch.nn.functional as nnf
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from src.train.model.albert_ner import *
from src.train.model.utils import *

# tqdm.pandas()


BASE_MODEL_PATH= './'
MODEL_PATH = "pytorch_model.bin"



def get_attribute_dict(df, attr_list):
    out = {}
    for attr in attr_list:
        out[attr] = []
        idx_list = []
        for i, lbl in enumerate(df['label']):
            if attr in lbl:
                if lbl == 'B-' + str(attr):
                    idx_list.append(i)
                if lbl == 'I-' + str(attr):
                    idx_list.append(i)
        for i in idx_list:
            if str(df['word'][i]).strip() != 'nan':
                out[attr].append(str(df['word'][i]))
    out = {k: ' '.join(v) for k, v in out.items() if v}
    if len(out) == 0:
        df['attributes'] = None
    else:
        df['attributes'] = out
    return df


def get_attributes(df):
    output_attr = []
    for _, attr, _ in df['ner_prediction']:
        output_attr.append(attr)
    df['output_attr'] = list(output_attr)

    if str(df['attributes']).strip() == 'nan' or df['attributes'] is None:
        df['target_attr'] = []
    else:
        target_attr = []
        df['attributes'] = eval(str(df['attributes']))
        for x in df['attributes'].keys():
            target_attr.append(x)
        # target_attr.remove('pattern_score')
        df['target_attr'] = list(target_attr)

    return df


def get_tokenized(tokenizer, text, device,job_args:JobArgs):
    ids = []
    inputs = tokenizer.encode(text, add_special_tokens=False)
    input_len = len(inputs)
    #     print(inputs)
    ids.extend(inputs)

    ids = ids[:job_args.model_configs.training_params.max_len - 2]
    ids = [2] + ids + [3]

    mask = [1] * len(ids)
    token_type_ids = [0] * len(ids)

    padding_len = job_args.model_configs.training_params.max_len - len(ids)

    ids = ids + ([0] * padding_len)
    mask = mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)

    return {
        "ids": torch.tensor([ids], dtype=torch.long).to(device, dtype=torch.long),
        "mask": torch.tensor([mask], dtype=torch.long).to(device, dtype=torch.long),
        "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long).to(device, dtype=torch.long),
    }


def get_tokenized_batch(tokenizer, queries, device,job_args:JobArgs):
    ids = []
    target_tag = []

    for i, s in enumerate(queries):
        inputs = tokenizer.encode(
            s,
            add_special_tokens=False
        )
        input_len = len(inputs)
        ids.extend(inputs)

    ids = ids[:job_args.model_configs.training_params.max_len  - 2]

    ids = [2] + ids + [3]

    mask = [1] * len(ids)
    token_type_ids = [0] * len(ids)

    padding_len = job_args.model_configs.training_params.max_len - len(ids)

    ids = ids + ([0] * padding_len)
    mask = mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)

    return {
        "ids": torch.tensor(ids, dtype=torch.long).to(device, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long).to(device, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long).to(device, dtype=torch.long)
    }


def predict(model, tokenizer, text, device, enc_tag,job_args: JobArgs):
    input_text = ' '.join(str(wrd) for wrd in text)
    input_ = get_tokenized(tokenizer, input_text, device,job_args)
    out = model(input_['ids'], input_['mask'], input_['token_type_ids'])
    # print(out)
    prob = nnf.softmax(out[0], dim=1)
    top_p, top_class = prob.topk(1, dim=1)
    top_p, top_class = top_p.detach().cpu().numpy().reshape(-1), top_class.detach().cpu().numpy().reshape(-1)
    tags = enc_tag.inverse_transform(top_class)

    tokens = tokenizer.convert_ids_to_tokens(input_['ids'][0])

    out = []

    tta = tags[0]
    tto = tokens[0]

    for i, j, k in zip(tokens, tags, top_p):
        if i not in ('[SEP]', '<pad>', '[CLS]'):
            if i[0] == '▁':
                # if len(out) > 0 and j != 'other' and j[0] == 'I' and j[2:] == out[-1][1][2:]:
                if len(out) > 0 and j == out[-1]:
                    out[-1] = [(out[-1][0] + ' ' + i.lstrip('▁')), out[-1][1], max(out[-1][2], k)]
                else:
                    out.append([i.lstrip('▁'), j, k])
            else:
                out[-1] = [(out[-1][0] + i), j if out[-1][1] == 'other' else out[-1][1], max(out[-1][2], k)]
    # return [(i[0], i[1][2:], i[2]) for i in out if i[1] != "other"]
    return [(i[1]) for i in out]


def predict_batch(model, tokenizer, queries, device, enc_tag,job_args: JobArgs):
    input_ = get_tokenized_batch(tokenizer, queries, device,job_args)

    outs = model(input_['ids'], input_['mask'], input_['token_type_ids'])
    # print(out)
    result = []
    for out in outs:
        prob = nnf.softmax(out[0], dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        top_p, top_class = top_p.detach().cpu().numpy().reshape(-1), top_class.detach().cpu().numpy().reshape(-1)
        tags = enc_tag.inverse_transform(top_class)

        tokens = tokenizer.convert_ids_to_tokens(input_['ids'][0])

        out = []

        tta = tags[0]
        tto = tokens[0]

        for i, j, k in zip(tokens, tags, top_p):
            if i not in ('[SEP]', '<pad>', '[CLS]'):
                if i[0] == '▁':
                    if len(out) > 0 and j != 'other' and j[0] == 'I' and j[2:] == out[-1][1][2:]:
                        out[-1] = [(out[-1][0] + ' ' + i.lstrip('▁')), out[-1][1], max(out[-1][2], k)]
                    else:
                        out.append([i.lstrip('▁'), j, k])
                else:
                    out[-1] = [(out[-1][0] + i), j if out[-1][1] == 'other' else out[-1][1], max(out[-1][2], k)]
        result.append([(i[0], i[1][2:], i[2]) for i in out if i[1] != "other"])


# def predict_in_batch(model, tokenizer, df, device, enc_tag, batch=8,job_args:JobArgs):
#     model.eval()
#     queries = df['word'].values
#     length = len(queries)
#     start = 0
#     end = min(batch, len(queries))
#     results = []

#     train_dataset = InferenceDataset(texts=queries, job_args, TOKENIZER=tokenizer)
#     data_loader = torch.utils.data.DataLoader(train_dataset, batch_size= job_args.model_configs.params.batch_size,
#                                               num_workers=1)

#     for _, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
#         ids = data['ids'].to(device, dtype=torch.long)
#         mask = data['mask'].to(device, dtype=torch.long)
#         token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
#         with torch.no_grad():
#             outputs = model(ids, mask, token_type_ids)

#         for out, input_ in zip(outputs, data['ids']):
#             prob = nnf.softmax(out, dim=1)
#             top_p, top_class = prob.topk(1, dim=1)
#             top_p, top_class = top_p.detach().cpu().numpy().reshape(-1), top_class.detach().cpu().numpy().reshape(-1)
#             tags = enc_tag.inverse_transform(top_class)

#             # tokens = tokenizer.convert_ids_to_tokens(input_[0])
#             tokens = tokenizer.convert_ids_to_tokens(input_)

#             out = []

#             for i, j, k in zip(tokens, tags, top_p):
#                 if i not in ('[SEP]', '<pad>', '[CLS]'):
#                     if i[0] == '▁':
#                         # if len(out) > 0 and j != 'other' and j[0] == 'I' and j[2:] == out[-1][1][2:]:
#                         if len(out) > 0 and j == out[-1]:
#                             out[-1] = [(out[-1][0] + ' ' + i.lstrip('▁')), out[-1][1], max(out[-1][2], k)]
#                         else:
#                             out.append([i.lstrip('▁'), j, k])
#                     else:
#                         out[-1] = [(out[-1][0] + i), j if out[-1][1] == 'other' else out[-1][1], max(out[-1][2], k)]
#             results.append([(i[1]) for i in out])
#         torch.cuda.empty_cache()
#     df['ner_prediction_batch'] = results
#     return df


def batch_predict(model, tokenizer, qdf, device, enc_tag,job_args:JobArgs):
    batch_size=job_args.model_configs.training_params.batch_size
    if qdf.shape[0] % batch_size == 0:
        num_batch = qdf.shape[0] / batch_size
    else:
        num_batch = math.floor((qdf.shape[0] / batch_size) + 1)

    result = []
    for n in range(int(num_batch)):
        start_idx = n * batch_size
        end_idx = start_idx + batch_size - 1
        batch_df = qdf.loc[start_idx: end_idx]
        queries = batch_df['word'].values.tolist()

        ids = tokenizer.batch_encode_plus(queries, add_special_tokens=False, is_split_into_words=True)
        ids = ids.get('input_ids')
        ids = [i[:job_args.model_configs.training_params.max_len  - 2] for i in ids]
        ids = [[2] + i + [3] for i in ids]
        mask = [[1] * len(i) for i in ids]
        token_type_ids = [[0] * len(i) for i in ids]

        ids = [i + ([0] * (job_args.model_configs.training_params.max_len  - len(i))) for i in ids]
        mask = [i + ([0] * (job_args.model_configs.training_params.max_len - len(i))) for i in mask]
        token_type_ids = [i + ([0] * (job_args.model_configs.training_params.max_len - len(i))) for i in token_type_ids]

        input_ = {
            "ids": torch.tensor(ids, dtype=torch.long).to(device, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long).to(device, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long).to(device, dtype=torch.long),
        }

        output = model(input_['ids'], input_['mask'], input_['token_type_ids'])

        prob = nnf.softmax(output, dim=2)
        top_p, top_class = prob.topk(1, dim=2)
        top_p = [i.detach().cpu().numpy().reshape(-1) for i in top_p]
        top_class = [i.detach().cpu().numpy().reshape(-1) for i in top_class]

        tags = [enc_tag.inverse_transform(i) for i in top_class]
        tokens = [tokenizer.convert_ids_to_tokens(i) for i in input_['ids']]

        for l, m, n in zip(tokens, tags, top_p):
            b_out = []
            for i, j, k in zip(l, m, n):
                if i not in ('[SEP]', '<pad>', '[CLS]'):
                    if i[0] == '▁':
                        # if len(out) > 0 and j != 'other' and j[0] == 'I' and j[2:] == out[-1][1][2:]:
                        if len(b_out) > 0 and j == b_out[-1]:
                            b_out[-1] = [(b_out[-1][0] + ' ' + i.lstrip('▁')), b_out[-1][1], max(out[-1][2], k)]
                        else:
                            b_out.append([i.lstrip('▁'), j, k])
                    else:
                        b_out[-1] = [(b_out[-1][0] + i), j if b_out[-1][1] == 'other' else b_out[-1][1],
                                     max(b_out[-1][2], k)]

            result.append([(i[1]) for i in b_out])

    qdf['ner_prediction_batch'] = result
    return qdf

    # for i in tqdm(range(ceil(length / batch)), total=ceil(length / batch)):
    #     l1 = queries[start:end]
    #
    #     batch_output = predict_batch(model, tokenizer, l1, device, enc_tag)
    #
    #     results.append(batch_output)
    #     if end >= length:
    #         break
    #     start = start + 128
    #     end = end + 128
    #     if end > length:
    #         end = length
    # results = sum(results, [])

    # return results


class NERTrainModelJob():
    def __init__(self, job_args:JobArgs) -> None:
        self.job_args = job_args

    def testing_run(self):
        

        if (torch.cuda.is_available()):
            device_num = torch.cuda.current_device()
            torch.cuda.memory_stats(device=device_num)

        # #device = torch.device("cuda")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info('testing model on golden dataset')
        input_path= "./"
        output_path = "./"
        
        temp = torch.load(output_path + "ner_albert_model.dt",map_location = torch.device('cpu'))
        model = EntityModel(len(temp['out_tag.bias']), job_args = self.job_args)

        model.load_state_dict(temp)
        model.to(device)
        enc_tag = joblib.load(output_path + "ner_enc_tag.bin")
        tokenizer = joblib.load(output_path + 'ner_tokenizer.bin')

        self.generate_model_matrix(model, tokenizer, device, enc_tag, input_path, output_path, test_data = None)

    def generate_model_matrix(self, model, tokenizer, device, enc_tag, input_path, output_path, test_data):
        if self.job_args.test_run:
            logger.info('Predicting attributes by ner model ')
            logger.info(" predicting on golden_dataset")
            df = pd.read_csv(input_path + 'gold_standard_dataset.csv', sep=',')
            df = df.dropna()
            df = df.groupby('query').agg({'word': list, 'label': list}).reset_index()
        else:
            logger.info('Predicting attributes by ner model ')
            if os.path.isfile(input_path + 'gold_standard_dataset.csv'):
                    logger.info(" predicting on golden_dataset")
                    df = pd.read_csv(input_path + 'gold_standard_dataset.csv', sep=',')
                    df = df.dropna()
                    df = df.groupby('query').agg({'word': list, 'label': list}).reset_index()
            else:
                if self.job_args.model_configs.incremental:
                    logger.info(" predicting on incremental_dataset")
                    df = pd.read_csv(input_path + 'incremental_dataset.csv', sep=',')
                else:
                    logger.info("predicting on ner_train_data_cs_gs dataset")
                    # df = pd.read_csv(input_path + 'ner_train_data_cs_gs.csv', sep='\t')
                    df = test_data
            
                df = df.dropna()
                df = df.groupby('query').agg({'word': list, 'label': list}).reset_index()
        # # generate_gold_ds.run_gen_gds(input_path,input_path + 'ner_gold_standard_dataset.csv')  #this transforms it to usable format

        df['ner_prediction'] = df.apply(
            lambda x: predict(model=model, tokenizer=tokenizer, text=x['word'], device=device, enc_tag=enc_tag,job_args=self.job_args),
            axis=1)

        df = batch_predict(model=model, tokenizer=tokenizer, qdf=df, device=device, enc_tag=enc_tag,job_args=self.job_args)
        logger.info('Predicting attributes successful')

        attr_list = list(enc_tag.classes_)

        df_pred = pd.DataFrame()
        df_true = pd.DataFrame()
        for label in attr_list:
            df_pred[label] = [1 if label in i else 0 for i in df['ner_prediction']]
            df_true[label] = [1 if label in i else 0 for i in df['label']]

        mcm = multilabel_confusion_matrix(np.array(df_true), np.array(df_pred))
        mcm = mcm.flatten().reshape(len(attr_list), 4)
        mcm = pd.DataFrame(data=mcm, columns=['TN', 'FP', 'FN', 'TP'], index=attr_list)
        mult_conf_matrix_file = f'{output_path}confusion_matrix.csv'
        mcm.to_csv(mult_conf_matrix_file)
        logger.info('Multilabel Confusion Matrix saved')

        cl_rep = classification_report(np.array(df_true), np.array(df_pred), target_names=attr_list, output_dict=True, zero_division=0)
        cl_rep = pd.DataFrame.from_dict(cl_rep).transpose()
        cl_rep_file = f'{output_path}classification_report.csv'
        cl_rep.to_csv(cl_rep_file)
        logger.info('Classification report saved')

        compare_list = [(query, word, i, j) for query, word, i, j in
                        zip(df['query'], df['word'], df['label'], df['ner_prediction']) if i != j]
        compare_frame = pd.DataFrame(data=compare_list, columns=['query', 'word', 'label', 'ner_prediction'])
        comp_list_file = f'{output_path}prediction.csv'
        compare_frame.to_csv(comp_list_file, index=False)

        compare_list_batch = [(query, word, i, j) for query, word, i, j in
                              zip(df['query'], df['word'], df['label'], df['ner_prediction_batch']) if i != j]
        compare_frame_batch = pd.DataFrame(data=compare_list_batch,
                                           columns=['query', 'word', 'label', 'ner_prediction_batch'])
        comp_list_file_batch = f'{output_path}ner_prediction_difference_batch.csv'
        compare_frame_batch.to_csv(comp_list_file_batch, index=False)

        logger.info('Difference in true and predicted attributes saved')

    def train(self, job_args: JobArgs):
        # logger.info("Downloading required files")

        tokenizer_local_path = "./"
        input_path = "./"
        output_path = "./"
        if (torch.cuda.is_available()):
            device_num = torch.cuda.current_device()
            torch.cuda.memory_stats(device=device_num)

        # #device = torch.device("cuda")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if job_args.model_configs.incremental:
            logger.info("Incremental Training:True.. Starting Incremental Training.")
            input_files = [os.path.join("./incremental/", i.file_name) for i in sorted(job_args.inc_datasets, key=lambda x: x.updated_at, reverse=True)]
            df, enc_tag = process_data(input_files, job_args)

            train = df
            test = df.sample(frac=job_args.model_configs.training_params.test_dataset_frac, random_state=20)
        else:
            logger.info("Incremental Training:False.. Starting Training from scratch.")
            df, enc_tag = process_data([input_path + 'ner_train_data_cs_gs.csv'], job_args)
            joblib.dump(enc_tag, output_path + "ner_enc_tag.bin")

            train = df.sample(frac=1 - (job_args.model_configs.training_params.test_dataset_frac), random_state=20)
            test = df.drop(index=train.index)


        sentences, tag = train["word"].values.tolist(), train["tag"].values.tolist()
        sentences_test, tag_test = test["word"].values.tolist(), test["tag"].values.tolist()

        if job_args.model_configs.incremental:
            TOKENIZER = joblib.load(tokenizer_local_path+'ner_tokenizer.bin')
        else:
            TOKENIZER = AlbertTokenizerFast.from_pretrained(tokenizer_local_path)

        joblib.dump(TOKENIZER, output_path + 'ner_tokenizer.bin')

        train_dataset = EntityDataset(texts=sentences, tags=tag, job_args=job_args, TOKENIZER=TOKENIZER,
                                    other_token=enc_tag.classes_.tolist().index('other'))

        train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=job_args.model_configs.training_params.batch_size, num_workers=0)

        test_dataset = EntityDataset(texts=sentences_test, tags=tag_test, job_args=job_args, TOKENIZER=TOKENIZER,
                                    other_token=enc_tag.classes_.tolist().index('other'))

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=job_args.model_configs.training_params.batch_size, num_workers=0)

        if job_args.model_configs.incremental:
            model_name = "ner_albert_model.dt"
            model_path = "./"
            model_dict = torch.load(model_path + "/" + model_name, map_location=torch.device('cpu'))
            model = EntityModel(len(model_dict['out_tag.weight']), job_args=job_args)
            model.load_state_dict(model_dict)
        else:
            model = EntityModel(num_tag = len(list(enc_tag.classes_)), job_args=job_args)

        model.to(device)

        optimizer, scheduler = model.get_optimizer(sentences, model)

        logger.info('NER Model Training Started')
        logger.info(f"Training batch size : {job_args.model_configs.training_params.batch_size}")
        logger.info(f"Epochs : {job_args.model_configs.training_params.epoch}")

        del df
        del train
        del sentences
        del train_dataset         
        del test_dataset         
        del TOKENIZER

        model = train_ner_model(model, optimizer, scheduler, train_data_loader,
                                job_args.model_configs.training_params.epoch, 0,
                                output_path)                                                                                      

        logger.info('NER Model Training Successful ')
        # torch.save(model.state_dict(), output_path + "ner_albert_model.dt")
        del model

        # Generate Confusion Matrix for Generated Suggestion Data
        temp = torch.load(output_path + "ner_albert_model.dt")

        model = EntityModel(len(temp['out_tag.bias']),job_args=job_args)
        model.load_state_dict(temp)
        model.to(device)

        # model = joblib.load(output_path + 'model.bin')
        enc_tag = joblib.load(output_path + "ner_enc_tag.bin")
        tokenizer = joblib.load(output_path + 'ner_tokenizer.bin')

        self.generate_model_matrix(model, tokenizer, device, enc_tag, input_path, output_path, test)

        logger.info('Uploading Model Data and Reports Done!')