import numpy as np
from bson import Base64
import json


def tranferImageToTag(image_data):
    """trander encoder

    Args:
        image_data (_type_): _description_
    """
    data_tranfer = Base64(image_data)
    
    return data_tranfer

def trader_all(image_dict):
    """tranfer an container dict

    Args:
        image_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    image_container = []
    for item in image_dict:
        image_container.append(tranferImageToTag(item))
        
    return image_container
        