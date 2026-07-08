from typing import Optional, Dict, Any, Tuple, List, Union
from sane.model.autoencoder import SANEAutoEncoder
from sane.data.datasets import WindowedDataset

class DownstreamTask:
    r"""
    Base class for defining downstream tasks to be executed during the training of a `SANEAutoEncoder`.

    This class provides a framework for scheduling and executing downstream tasks (e.g., validation, testing, checkpointing)
    at specified epochs or frequencies during training. Subclasses must implement the `_execute` method to define
    the task-specific logic. If the various datasets are not provided, they will be taken from the trainer.

    Args:
        name (str): Name of the downstream task.
        epochs (Optional[Union[int, List[int]]]): Epochs at which the task should be executed. If an `int` is provided,
            it will be wrapped in a list. Default: ``None``.
        frequency (Optional[int]): Frequency (in epochs) at which the task should be executed. If both `epochs` and
            `frequency` are provided, `frequency` is ignored. If neither is provided, the task is executed every epoch.
            Default: ``None``.
        trainset (Optional[WindowedDataset]): Training dataset. Default: ``None``.
        valset (Optional[WindowedDataset]): Validation dataset. Default: ``None``.
        testset (Optional[WindowedDataset]): Test dataset. Default: ``None``.

    Note:
        If both `epochs` and `frequency` are provided, `frequency` is ignored.
        If neither is provided, the task is executed every epoch by default.
    """
    def __init__(
        self,
        name: str,
        epochs: Optional[Union[int, List[int]]] = None,
        frequency: Optional[int] = None,
        trainset: Optional[WindowedDataset] = None,
        valset: Optional[WindowedDataset] = None,
        testset: Optional[WindowedDataset] = None,
    ):
        self.name = name
        self.epochs = [epochs] if isinstance(epochs, int) else epochs
        self.frequency = frequency
        if self.epochs is not None and self.frequency is not None:
            self.frequency = None  # Disable frequency if epochs are specified
        if self.epochs is None and self.frequency is None:
            self.frequency = 1  # Default frequency if neither is specified
        self.trainset = trainset
        self.valset = valset
        self.testset = testset

    def __call__(self, sane_ae: SANEAutoEncoder, epoch: int, *args, **kwargs) -> Optional[Any]:
        r"""
        Executes the downstream task if the current epoch matches the specified schedule.

        Args:
            sane_ae (SANEAutoEncoder): The autoencoder model.
            epoch (int): Current training epoch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[Any]: The result of the task execution, or ``None`` if the task is not scheduled for this epoch.
        """
        if (self.epochs is not None and epoch in self.epochs) or (self.frequency is not None and epoch % self.frequency == 0):
            return self._execute(sane_ae, *args, epoch=epoch, **kwargs)
        else:
            return None

    def _execute(self, sane_ae: SANEAutoEncoder, *args, **kwargs) -> Any:
        r"""
        Implements the task-specific logic. Must be overridden by subclasses.

        Args:
            sane_ae (SANEAutoEncoder): The autoencoder model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")