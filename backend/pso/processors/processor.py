from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Processor(ABC, Generic[T]):
    """An abstract base class for processors that process data of type T."""

    @abstractmethod
    def process(self, data: T) -> T:
        """Processes the given data and returns the processed data.

        Args:
            data (T): The data to be processed.

        Returns:
            T: The processed data.
        """
        pass
