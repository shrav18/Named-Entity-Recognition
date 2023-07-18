from inviz.clients.schemas.mlops import AttributeType
from src.schema.schema import AttributeConfig
from typing import Dict
import json 

def field_map(att_configs,default_attributes):
    global_names={}
    output={}

    for att in att_configs:
        global_names[att.field_name] = att.name
    for i,j in default_attributes.grouping_fields.items():
        name=global_names[i]
        for att in j:
            output[att] = name
    json.dump(output, open('fieldmap.json', 'w'))
    return output
        
def attribute_config(txt) -> (dict, Dict[str, AttributeConfig]):

        attribute_map: Dict[str, AttributeConfig] = {}
        for attribute in txt:
            if attribute.attribute_type != AttributeType.CLS:
                attribute_map[attribute.name] = attribute
        return attribute_map