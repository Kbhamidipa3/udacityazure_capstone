# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Column1": pd.Series([0], dtype="int64"), "Casino": pd.Series([False], dtype="bool"), "Free_internet": pd.Series([False], dtype="bool"), "Gym": pd.Series([False], dtype="bool"), "Helpful_votes": pd.Series([0], dtype="int64"), "Hotel_name": pd.Series(["example_value"], dtype="object"), "Hotel_stars": pd.Series([0], dtype="int64"), "Member_years": pd.Series([0.0], dtype="float64"), "Nr_hotel_reviews": pd.Series([0], dtype="int64"), "Nr_reviews": pd.Series([0], dtype="int64"), "Nr_rooms": pd.Series([0.0], dtype="float64"), "Period_of_stay": pd.Series(["example_value"], dtype="object"), "Pool": pd.Series([False], dtype="bool"), "Review_month": pd.Series(["example_value"], dtype="object"), "Review_weekday": pd.Series(["example_value"], dtype="object"), "Spa": pd.Series([False], dtype="bool"), "Tennis_court": pd.Series([False], dtype="bool"), "Traveler_type": pd.Series(["example_value"], dtype="object"), "User_continent": pd.Series(["example_value"], dtype="object"), "User_country": pd.Series(["example_value"], dtype="object")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
# You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
