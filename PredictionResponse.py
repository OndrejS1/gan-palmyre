import json
from dataclasses import dataclass


@dataclass
class PredictionResponse:
    predicted_class: str
    probability: float

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
