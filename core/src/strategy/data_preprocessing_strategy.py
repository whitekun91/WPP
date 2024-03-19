from abc import *

class DataPreprocessingStrategy(ABC):

    @abstractmethod
    def run_algorithm(self):
        pass