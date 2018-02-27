from typing import List, Dict

import numpy as np
import sys

from core.Prediction import Prediction


class TimeSeries:
    def __init__(self, **kwargs):
        self.predictions = []  # List[Prediction]
        self.__parse_kwargs__(**kwargs)

    def __parse_kwargs__(self, **kwargs):
        def __extract_values__(values):
            if isinstance(values, str):
                return [float(x) for x in values.split(" ")]
            elif isinstance(values, List):
                return [float(x) for x in values]
            else:
                RuntimeError("Values class not recognized. {}".format(values))
                sys.exit(-2)

        self.forecasting_window = int(kwargs.pop("forecasting_window", 1))
        self.minimum_observations = int(kwargs.pop("minimum_observations", 1))

        self.observations = __extract_values__(kwargs.pop("observations", []))

        predicted_values = __extract_values__(kwargs.pop("predicted_values", []))
        predicted_stddev = __extract_values__(kwargs.pop("predicted_stddev", []))
        assert len(predicted_values) == len(predicted_stddev), \
            "The length of predicted values ({}) and of the stddev ({}) are not equal.".format(
                len(predicted_values), len(predicted_stddev))
        self.predictions = [Prediction(predicted_values[i], predicted_stddev[i])
                            for i in np.arange(len(predicted_values))]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_csv(self) -> Dict:
        assert isinstance(self.predictions, List[Prediction]), "prediction attribute must be of type List[Prediction]"
        return {
            "observations": " ".join(str(x) for x in self.observations),
            "predicted_values": " ".join(str(x.value) for x in self.predictions),
            "predicted_stddev": " ".join(str(x.stddev) for x in self.predictions),
            **{key: self.__getattribute__(key) for key in self.__dict__.keys() - {"observations", "predictions"}}
        }
