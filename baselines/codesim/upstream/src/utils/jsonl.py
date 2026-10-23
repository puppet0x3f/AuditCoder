import json
import os

# Read an jsonl file and convert it into a python list of dictionaries.
def read_jsonl(filename):
    """Reads a jsonl file and yields each line as a dictionary"""
    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lines.append(json.loads(line))
    return lines

# Write a python list of dictionaries into a jsonl file
def write_jsonl(filename, lines):
    """Writes a python list of dictionaries into a jsonl file"""
    
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")


def append_in_jsonl(filename, line):
    """Appends a python dictionaries into a jsonl file"""
    
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, mode="a", encoding="utf-8") as file:
        file.write(json.dumps(line) + "\n")

