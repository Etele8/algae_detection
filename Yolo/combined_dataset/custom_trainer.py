from ultralytics.engine.trainer import BaseTrainer
from custom_dataset import TwoChannelDataset
import yaml

class CustomTrainer(BaseTrainer):
    def get_dataset(self):
        print("Using custom dataset loader...")
        with open(self.args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        return TwoChannelDataset(
            img_path=data_config['train'],  # Use the train path from data.yaml
            imgsz=self.args.imgsz,
            data=data_config
        )