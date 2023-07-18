import os.path

from inviz.core.environment import EnvStorage
from inviz.clients.schemas.mlops import ModelType
from inviz.dataset.file_path_helper import FilesHelper


class PathUtils(FilesHelper):

    def __init__(self, app_id: str, model_type: ModelType = ModelType.NER) -> None:
        super().__init__(app_id, model_type)

    def get_ner_dataprep_output_path(self, model_id):
        return os.path.join(EnvStorage.BASE_DIR, self.app_id, 'models', self.model_type, model_id, 'preprocessing')
