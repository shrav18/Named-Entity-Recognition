import argparse
import asyncio
import os

from datetime import datetime
import aiohttp
from zipfile import ZipFile
from typing import Dict
from loguru import logger
import glob
import pandas as pd

from src.core.config import   APP_ID, MODEL_ID,TEST_RUN
from src.train.ner_train_model import NERTrainModelJob   #change
from src.schema.schema import JobArgs
#mlops
from inviz import *
from inviz.clients.aioclient import aio_client
from inviz.clients.mlops import MLOpsClient   #done
from inviz.clients.schemas.mlops import ModelStatus, DatasetType
from inviz.storage import cloud_store
from inviz.clients.authentication import AuthenticationClient
from inviz.core.environment import EnvIO,EnvStorage
from inviz.clients.schemas.mlops import ModelType,UpdateModelStatus,ModelDetails
from src.storage.path_utils import PathUtils


def download_input_files(job_args: JobArgs):
    path_utils=PathUtils(job_args.app_id)
    path = path_utils.get_albert_model_path(job_args.base_model_config.id)
    cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path=path_utils.ALBERT_MODEL)
    with open(path_utils.ALBERT_MODEL, "rb") as f:
        zip_ref = ZipFile(f)
        zip_ref.extractall()
    
    path = path_utils.get_golden_dataset_path(None)
    cloud_store.storage.download_file(EnvStorage.BUCKET, path,local_path=path_utils.TESTING_DATASET)

    if job_args.model_configs.incremental:
        if not os.path.exists("./incremental"):
            os.makedirs("./incremental")
        for dataset in job_args.inc_datasets:
            path = path_utils.get_incremental_dataset_path(dataset.file_name)
            cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = os.path.join("./incremental/", dataset.file_name))

        # path = path_utils.get_ner_prediction_path(job_args.model_configs.incremental_model_id)
        # cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = path_utils.NER_PREDICTION_FILE_NAME)

        path = path_utils.get_ner_model_path(job_args.model_configs.incremental_model_id)
        cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = path_utils.NER_TRAINED_MODEL_PATH)
        with open(path_utils.NER_TRAINED_MODEL_PATH, "rb") as f:
            zip_ref = ZipFile(f)
            zip_ref.extractall()

    elif job_args.test_run:
        path = path_utils.get_ner_model_path(job_args.model_id)
        cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = path_utils.NER_TRAINED_MODEL_PATH)
        with open(path_utils.NER_TRAINED_MODEL_PATH, "rb") as f:
            zip_ref = ZipFile(f)
            zip_ref.extractall()
    else:
        path = path_utils.get_ner_train_data_path(job_args.model_id)
        cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = path_utils.NER_TRAIN_DATA_FILE_NAME)

        if job_args.model_configs.use_uploaded_dataset:
            cloud_store.storage.download_file(EnvStorage.BUCKET, path_utils.get_uploaded_dataset_path(), path_utils.UPLOADED_DATASET)
            df1 = pd.read_csv('ner_train_data_cs_gs.csv')
            df2 = pd.read_csv('training_dataset.csv')
            combined_df = df1.append(df2).drop_duplicates(subset=['query','word','label'], keep='last').sort_values(['query','word','label'])
            combined_df.to_csv("ner_train_data_cs_gs.csv", index=False)


def upload_output_files(job_args: JobArgs):
    path_utils = PathUtils(job_args.app_id)
    report_files = ['classification_report.csv', 'confusion_matrix.csv',
                    'prediction.csv', 'ner_prediction_difference_batch.csv']
    for file in report_files:
        cloud_store.storage.upload_file(EnvStorage.BUCKET, file,
                                    os.path.join(path_utils.get_ner_train_output_path(job_args.model_id), 'testing', file))

    if not job_args.test_run:
        model_files = ["ner_enc_tag.bin", "ner_tokenizer.bin", "ner_albert_model.dt"]
        with ZipFile('ner_model.zip', 'w') as zip_object:
        # Adding files that need to be zipped
            for file in model_files:
                zip_object.write(file)
        cloud_store.storage.upload_file(EnvStorage.BUCKET, path_utils.NER_TRAINED_MODEL_PATH,
                                    os.path.join(path_utils.get_ner_train_output_path(job_args.model_id), 'training', path_utils.NER_TRAINED_MODEL_PATH))

    if job_args.model_configs.incremental:
        src_path = os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_configs.incremental_model_id), "facet_value_map.json")
        dest_path = os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_configs.id), "facet_value_map.json")
        cloud_store.storage.copy_object(EnvStorage.BUCKET, src_path, EnvStorage.BUCKET, dest_path)

        src_path = os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_configs.incremental_model_id), "ner_attributes_map.json")
        dest_path = os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_configs.id), "ner_attributes_map.json")
        cloud_store.storage.copy_object(EnvStorage.BUCKET, src_path, EnvStorage.BUCKET, dest_path)

def update_status(job_args: JobArgs, status: ModelStatus, message: str):
    logger.info(f'setting model config to {status}')
    temp = UpdateModelStatus(model_id = job_args.model_id, status= status, status_message = message)
    MLOpsClient().update_model_status(job_args.app_id, temp, job_args.jwt_token)

def run_modeltrain(job_args: JobArgs):
    try:
        # Init storage provider
        cloud_store.initiate_storage(EnvStorage.PROVIDER)

        job_args.jwt_token = AuthenticationClient().get_service_auth_token()

        update_status(job_args, ModelStatus.TRAINING, "model is in training")

        # Fetch all required configurations
        logger.info('Fetching all required configurations')
        model_configs = MLOpsClient().get_model_configs(job_args.app_id, job_args.model_id, ModelType.NER,
                                                        job_args.jwt_token)
        if len(model_configs) == 0:
            raise Exception("Model config does not found")

        job_args.model_configs = model_configs[0]

        job_args.base_model_config = MLOpsClient().get_active_model(job_args.app_id, ModelType.DOMAIN_ALBERT,
                                                                    job_args.jwt_token, ModelDetails)

        if job_args.model_configs.incremental:
            job_args.inc_datasets = MLOpsClient(auth_token=job_args.jwt_token).get_datasets(
                job_args.app_id, DatasetType.INC_TRAINING, ModelType.NER,
                job_args.model_configs.incremental_dataset_ids)

        # Download required files
        download_input_files(job_args)

        trainer = NERTrainModelJob(job_args)
        if job_args.test_run:
            #test model
            trainer.testing_run()
        else:
            # train dataset
            trainer.train(job_args)

        # Upload output files
        upload_output_files(job_args)

        update_status(job_args, ModelStatus.TRAINED, "model training is done")

    except Exception as e:
        update_status(job_args, ModelStatus.TRAINING_FAILED, "model training is failed")
        logger.error("Error: ", exc_info=e)
        raise e


def main():
    PARSER = argparse.ArgumentParser(description='Run a PySpark job')
    PARSER.add_argument('--APP_ID', type=str, required=False, dest="app_id", default=APP_ID, help="")
    PARSER.add_argument('--MODEL_ID', type=str, required=False, dest="model_id", default=MODEL_ID,
                        help="")
    PARSER.add_argument("--TEST_RUN", required=False, dest="test_run", action="store_true", default=TEST_RUN, help="")

    job_args: JobArgs = JobArgs(**PARSER.parse_args().__dict__)

    logger.info(job_args.__dict__)

    if job_args.app_id is None or job_args.model_id is None:
        raise Exception('Required arguments/environment variables: APP_ID, MODEL_ID')

    start_time = datetime.now()
    logger.info(' Starting job at {}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))


    run_modeltrain(job_args)

    logger.info("Time taken : " + str(datetime.now() - start_time))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("Error: ", exc_info=e)
        raise e




