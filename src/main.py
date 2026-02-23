from src.data.loader import WISDMSplits
from src.config import conf
from src.model.describe_net import DescribeNetWISDM
from src.training.train import train_model


def main():
    # Load data
    data = WISDMSplits(
        base_path=conf.wisdm_data_files,
        sensor_device=conf.sensor_device,
        signal_type=conf.signal_type,
        window=20,
        step=10,
        test_size=0.1,
        val_size=0.2,
        seed=42,
        activity_code="A",
    )

    # Model
    model = DescribeNetWISDM(output_dim=len(conf.subject_ids))

    # Train on train split, validate on val split
    train_model(
        model=model,
        train_features=data.train_segments,
        train_labels=data.train_labels,
        val_features=data.val_segments,
        val_labels=data.val_labels,
        nr_of_epochs=10,
        batch_size=64,
    )


if __name__ == "__main__":
    main()
