import os.path

from inviz.core.environment import EnvStorage
from inviz.clients.schemas.mlops import ModelType
from inviz.dataset.file_path_helper import FilesHelper

class PathUtils(FilesHelper):

    NER_TRAIN_DATA_FILE_NAME = "ner_train_data_cs_gs.csv"
    NER_TRAINED_MODEL_FILE_NAME = 'ner_albert_model.dt'
    NER_TRAINED_TOKENIZER = 'ner_tokenizer.bin'
    NER_ENC_TAG_FILE_NAME = 'ner_enc_tag.bin'
    NER_TRAINED_MODEL_PATH = 'ner_model.zip'
    NER_PREDICTION_FILE_NAME = 'prediction.csv'


    def __init__(self, app_id: str, model_type: ModelType = ModelType.NER) -> None:
        super().__init__(app_id, model_type)

    def get_ner_train_data_path(self, model_id):
        return os.path.join(
            EnvStorage.BASE_DIR, self.app_id, "models", self.model_type, model_id, "preprocessing", self.NER_TRAIN_DATA_FILE_NAME)

    def get_ner_trained_model_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type ,model_id, 'training', self.NER_TRAINED_MODEL_FILE_NAME)

    def get_ner_trained_tokenizer_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type, model_id, 'training', self.NER_TRAINED_TOKENIZER)

    def get_ner_enc_tag_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type, model_id, 'training', self.NER_ENC_TAG_FILE_NAME)

    def get_ner_prediction_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type, model_id,'testing', self.NER_PREDICTION_FILE_NAME)

    def get_ner_train_output_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type, model_id)

    def get_ner_model_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type, model_id, 'training',self.NER_TRAINED_MODEL_PATH)

    def get_ner_dataprep_output_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type, model_id, 'preprocessing')
