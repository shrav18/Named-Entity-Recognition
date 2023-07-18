import argparse
import asyncio
import os
import copy
import json
from datetime import datetime

import aiohttp
from loguru import logger

from src.core.config import  APP_ID, MODEL_ID
from src.dataprep.ner_train_dataprep import NERTrainDataGeneration
from src.schema.schema import JobArgs
#mlops
from inviz import *
from inviz.clients.aioclient import aio_client
from inviz.clients.mlops import MLOpsClient
from inviz.clients.schemas.mlops import ModelStatus
from inviz.storage import cloud_store
from inviz.clients.authentication import AuthenticationClient
from inviz.core.environment import EnvIO,EnvStorage
from inviz.clients.schemas.mlops import ModelType,UpdateModelStatus
from inviz.clients.app_config import AppConfigClient
from src.storage.path_utils import PathUtils
from src.resources.utils import field_map,attribute_config


def download_input_files(job_args: JobArgs):
    path_utils = PathUtils(job_args.app_id)
    path = path_utils.get_generated_suggestion_path()
    cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = path_utils.GENERATED_SUGGESTION_DATASET)
    path = path_utils.get_catalog_dataset_path()
    cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = path_utils.CATALOG_DATASET)
    path = path_utils.get_click_stream_queries_path()
    cloud_store.storage.download_file(EnvStorage.BUCKET, path, local_path = path_utils.CLICK_STREAM_DATASET)


def upload_output_files(job_args: JobArgs):
    path_utils = PathUtils(job_args.app_id)
    cloud_store.storage.upload_file(EnvStorage.BUCKET, 'ner_attributes_map.json',
                                  os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_id), 'ner_attributes_map.json'))
    cloud_store.storage.upload_file(EnvStorage.BUCKET, 'facet_value_map.json',
                                  os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_id), 'facet_value_map.json'))
    cloud_store.storage.upload_file(EnvStorage.BUCKET, 'ner_train_data_cs.csv',
                                  os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_id), 'ner_train_data_cs.csv'))
    cloud_store.storage.upload_file(EnvStorage.BUCKET, 'ner_train_data_cs_gs.csv',
                                  os.path.join(path_utils.get_ner_dataprep_output_path(job_args.model_id), 'ner_train_data_cs_gs.csv'))

def update_status(job_args: JobArgs,status:ModelStatus,message:str):
    logger.info(f'setting model config to {status}')
    temp = UpdateModelStatus(model_id = job_args.model_id, status= status, status_message = message)
    MLOpsClient().update_model_status(job_args.app_id, temp, job_args.jwt_token)

def run_dataprep(job_args: JobArgs):
    # Init storage provider
    cloud_store.initiate_storage(EnvStorage.PROVIDER)

    # Fetch all required configurations
    logger.info('Fetching all required configurations')
    job_args.jwt_token = AuthenticationClient().get_service_auth_token()
    attribute_conf = MLOpsClient().get_attributes_configs(job_args.app_id, None, job_args.jwt_token)
    default_search_conf = AppConfigClient().get_default_search_config(app_id = job_args.app_id,token = job_args.jwt_token)
    job_args.field_map = field_map(attribute_conf, default_search_conf )
    job_args.attr_configs = attribute_config(attribute_conf)
    job_args.model_configs = MLOpsClient().get_model_configs(job_args.app_id, job_args.model_id, ModelType.NER,
                                                                        job_args.jwt_token)
    update_status(job_args, ModelStatus.YET_TO_TRAIN, "NER dataprep is yet to start")

    try:
        update_status(job_args, ModelStatus.TRAINING, "NER dataprep is in process")

        # Download required files
        download_input_files(job_args)
        # Generate dataset
        generator = NERTrainDataGeneration()
        generator.generate(job_args)

        # Upload output files
        upload_output_files(job_args)

        update_status(job_args, ModelStatus.TRAINING,"NER dataprep is done")
    except Exception as e:
        update_status(job_args, ModelStatus.TRAINING_FAILED,"NER dataprep failed")
        logger.error("Error: ", exc_info=e)
        raise e


def main():
    PARSER = argparse.ArgumentParser(description='Run a PySpark job')
    PARSER.add_argument('--APP_ID', type=str, required=False, dest="app_id", default=APP_ID, help="")
    PARSER.add_argument('--MODEL_ID', type=str, required=False, dest="model_id", default=MODEL_ID, help="")

    job_args: JobArgs = JobArgs(**PARSER.parse_args().__dict__)

    logger.info(job_args.__dict__)

    if job_args.app_id is None or job_args.model_id is None:
        raise Exception('Required arguments/environment variables: APP_ID, MODEL_ID')

    start_time = datetime.now()
    logger.info(' Starting job at {}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))


    run_dataprep(job_args)

    logger.info("Time taken : " + str(datetime.now() - start_time))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("Error: ", exc_info=e)
        raise e
