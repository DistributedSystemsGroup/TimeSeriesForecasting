from typing import List, Dict

from core.Prediction import Prediction


class TimeSeries:
    def __init__(self, observations: List[float], **kwargs):
        self.observations = observations
        self.predictions = []  # List[Prediction]
        self.__parse_kwargs__(**kwargs)

    def __parse_kwargs__(self, **kwargs):
        self.forecasting_window = int(kwargs.pop("forecasting_window", 1))
        self.minimum_observations = int(kwargs.pop("minimum_observations", 1))
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
