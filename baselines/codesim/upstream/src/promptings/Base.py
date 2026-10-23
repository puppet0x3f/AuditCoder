from typing import List
import tiktoken
import os
import copy
import time

from models.Base import BaseModel
from datasets.Dataset import Dataset
from results.Results import Results
from utils.parse import parse_response
from time import perf_counter_ns
from constants.verboseType import *

class BaseStrategy(object):
    def __init__(
        self,
        model: BaseModel,
        data: Dataset,
        language: str,
        pass_at_k: int,
        results: Results,
        verbose: int = VERBOSE_FULL,
    ):
        self.model = model
        self.data = data
        self.pass_at_k = pass_at_k
        self.results = results
        self.language = language
        self.verbose = verbose
        self.run_details = []
    

    def append_run_details(self, run_details: dict):
        for key in run_details.keys():
            if key in self.run_details:
                self.run_details[key] += run_details[key]
            else:
                self.run_details[key] = run_details[key]


    def gpt_chat(
            self, 
            processed_input: List[dict], 
            frequency_penalty=0, 
            presence_penalty=0
        ):
        
        response, run_details = self.model.prompt(
            processed_input=processed_input, 
            frequency_penalty=frequency_penalty, 
            presence_penalty=presence_penalty
        )
        self.append_run_details(run_details)
        
        return response


    def run_single_pass(self, data_row: dict):
        pass

    def run(self, record_full_result):
        # self.data.data.reverse()
        
        num_items = len(self.data)
        num_success = 0

        for i, data_row in enumerate(self.data):
            if self.verbose >= VERBOSE_FULL:
                print("", flush=True, end="")

            found = False
            for j in range(len(self.results)):
                if self.results[j]["task_id"] == data_row[self.data.id_key]:
                    item = copy.deepcopy(self.results[j])
                    cur_pass = len(item["source_codes"])
                    is_solved = item["is_solved"]
                    cur_imp = item["source_codes"][-1]
                    found = True
                    break
            if not found:
                item = {
                    self.data.id_key: data_row[self.data.id_key],
                    "task_id": data_row[self.data.id_key],
                    "language": self.language,
                    "source_codes": [],
                    "run_details": [],
                    "no_of_try": 0,
                }

                cur_pass = 0
                is_solved = False
                cur_imp = ""

            while cur_pass < self.pass_at_k and not is_solved:
                # initialize it for each run
                self.run_details = {}
                # for _ in range(10):
                #     try:
                response = self.run_single_pass(data_row)
                #     break
                # except Exception as e:
                #     time.sleep(5)
                #     pass

                cur_imp = parse_response(response)

                item["source_codes"].append(cur_imp)

                # Remove Full details
                if not record_full_result:
                    del self.run_details["details"]

                item["run_details"].append(self.run_details)
                
                item["no_of_try"] += 1

                is_solved = self.data.evaluate(
                    item=data_row,
                    cur_imp=cur_imp,
                    language=self.language
                )

                cur_pass += 1
            
            if is_solved:
                num_success += 1

            item["is_solved"] = is_solved

            self.results.get_results().insert(i, item)

            # Deleting duplicate results
            k = i + 1
            while True:
                # Termination condition
                if k >= len(self.results):
                    break
                
                # Deleting duplicate results
                if self.results[k]["task_id"] == data_row[self.data.id_key]:
                    del self.results.results[k]
                
                # Increment
                k += 1

            if self.verbose >= VERBOSE_MINIMAL:
                print(f'completed {i+1}/{num_items}, Solved: {self.results[i]["is_solved"]}, number of success = {num_success}/{i+1}, acc = {round(num_success/(i+1)*100, 2)}')
            
            if not found:
                self.results.save_results()

            if self.verbose >= VERBOSE_FULL:
                print("", flush=True, end="")
          
        
        if len(self.results) > len(self.data):
            self.results.results = self.results[:len(self.data)]
            self.results.save_results()
