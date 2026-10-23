import logging
import sys

import traceback
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.sleep_time = 0


    @abstractmethod
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(self, processed_input, frequency_penalty=0, presence_penalty=0):
        pass

