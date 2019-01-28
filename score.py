import json
import numpy as np
import os
import tensorflow as tf
import keras
from azureml.core.model import Model
from azureml.core import Workspace

def init():
    global model
    modelName = 'kerasModel'
    workspaceName = 'mlworkspace'
    subscriptionId = '08cb84a2-122a-4813-a67e-88d3c2530ea1'
    resourceGroup = 'ml-openhack-rg'
    # retrieve the path to the model file using the model name
    ws = Workspace.get(name = workspaceName,
                       subscription_id = subscriptionId,    
                       resource_group = resourceGroup)
    
    model_path = Model.get_model_path(modelName, version=None, _workspace=ws)
    model = tf.keras.models.load_model(model_path)
    
def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    results = model.predict(data)
    return json.dumps(results.tolist())