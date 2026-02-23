import os
from enum import Enum
import kagglehub
import torch


class SensorDevice(Enum):
    PHONE = "phone"
    WATCH = "watch"


class SignalType(Enum):
    ACCEL = "accel"
    GYRO = "gyro"


class Config:

    def __init__(self):
        # set device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )

        # download datafiles + set path
        self.wisdm_data_files = os.path.join(
            kagglehub.dataset_download(
                "mashlyn/smartphone-and-smartwatch-activity-and-biometrics"
            ),
            "wisdm-dataset",
            "wisdm-dataset",
        )

        self.sensor_device = SensorDevice.PHONE.value
        self.signal_type = SignalType.GYRO.value

        self.subject_ids = [i for i in range(1600, 1651)]


conf = Config()
