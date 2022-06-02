import abc


class Named(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
