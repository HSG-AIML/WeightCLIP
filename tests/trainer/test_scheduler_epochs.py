from sane.data.datasets.cached_windowed_dataset import CachedWindowedDataset
from sane.trainer.trainer import SANETrainer


def _make_trainer(epochs: int, scheduler_epochs: int | None = None, num_batches: int = 5) -> SANETrainer:
    trainer = SANETrainer()
    trainer.config = {"epochs": epochs}
    if scheduler_epochs is not None:
        trainer.config["scheduler_epochs"] = scheduler_epochs
    trainer.trainset = CachedWindowedDataset.__new__(CachedWindowedDataset)
    trainer.trainloader = [object()] * num_batches
    return trainer


def test_get_total_steps_defaults_to_training_epochs() -> None:
    trainer = _make_trainer(epochs=3, num_batches=5)
    assert trainer._get_total_steps() == 15


def test_get_total_steps_uses_scheduler_epochs_override() -> None:
    trainer = _make_trainer(epochs=3, scheduler_epochs=10, num_batches=5)
    assert trainer._get_total_steps() == 50