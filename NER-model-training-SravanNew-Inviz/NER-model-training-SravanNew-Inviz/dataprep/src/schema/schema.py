from typing import Optional, List, Dict
from inviz.clients.schemas.mlops import ModelDetails, AttributeConfig

from pydantic import BaseModel, Field


class JobArgs(BaseModel):
    # Arguments
    app_id: str
    model_id: str
    model_configs: Optional[ModelDetails]
    field_map: Optional[dict]
    attr_configs: Optional[Dict[str, AttributeConfig]]
    jwt_token: Optional[str]