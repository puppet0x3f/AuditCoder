from typing import List
from .Dataset import Dataset
from evaluations.func_evaluate import evaluate_functional_correctness, evaluate_io
from constants.paths import *


class HumanDataset(Dataset):
    def __init__(
        self,
        path: str = HUMAN_DATA_PATH,
    ):
        super().__init__(path)
        self.id_key = "task_id"

    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        result = evaluate_functional_correctness(
            test=item["test"],
            entry_point=item["entry_point"],
            completion=cur_imp,
        )
        return result == "passed"

    def evaluate_sample_io(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        
        return evaluate_io(
            sample_io=item["sample_io"],
            completion=cur_imp,            
        )
    
    
    def evaluate_additional_io(
        self,
        id: int,
        io: List[str],
        cur_imp: str,
        language: str,
    ):
        if len(io) == 0:
            return True, ""
        
        return evaluate_io(
            sample_io=io,
            completion=cur_imp,            
        )

    @staticmethod
    def get_prompt(item):
        if "prompt" in item:
            return f"{item['prompt'].strip()}"
        elif "text" in item:
            return f"{item['text'].strip()}"
        else:
            raise Exception("No prompt or text in item")
