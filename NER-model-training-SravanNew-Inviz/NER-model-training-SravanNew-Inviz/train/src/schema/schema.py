from typing import Optional, List, Dict
from inviz.clients.schemas.mlops import ModelDetails, AttributeConfig, DatasetDetails

from pydantic import BaseModel, Field

class JobArgs(BaseModel):
    # Arguments
    app_id: str
    model_id: str
    test_run: bool
    model_configs: Optional[ModelDetails]
    base_model_config: Optional[ModelDetails]
    field_map: Optional[dict]
    attr_configs: Optional[Dict[str, AttributeConfig]]
    inc_datasets: Optional[List[DatasetDetails]]
    jwt_token: Optional[str]
