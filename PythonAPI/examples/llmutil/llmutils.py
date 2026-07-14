from __future__ import annotations

import logging
import os
import pathlib
import random
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import dotenv
import numpy as np
import openai
from openai import OpenAI
import pystache

from . import utils
from .utils import Action, Environment, State

log = logging.getLogger(__name__)

env_file = os.path.join(pathlib.Path(__file__).parent.parent.parent, ".env")
dotenv.load_dotenv(env_file, override=True)
openai_api_key = os.environ.get("OPENAI_KEY")
#openai_api_key = "" #or directly add here
openai_client = OpenAI(api_key=openai_api_key)

ENGINE = "gpt-5.4"
#ENGINE = "gpt-4o"
_LLM_POOL = ThreadPoolExecutor(max_workers=1)


def add_gaussian_noise(ground_plan: List[Action], std: float) -> List[Action]:
    noised_plan = []
    for action in ground_plan:
        noised_params = []
        for param in action.params:
            if isinstance(param, float) or isinstance(param, int):
                noised_params.append(param + np.random.normal(0, std, 1)[0])
            else:
                noised_params.append(param)

        noised_plan.append(Action(action.name, noised_params))
    return noised_plan


def guassian_rejection_sample(
    env: Environment,
    ground_plan: List[Action],
    max_noise: float = 1.0,
    max_attempts: int = 10000,
) -> Tuple[List[Action], int]:
    """A constraint satisfaction strategy that randomly samples input vectors
    until it finds one that satisfies the constraints.

    If none are found, it returns the most common mode of failure.

    This function also returns the number of CSP samples
    """
    violation_modes = Counter()
    for i in range(max_attempts):
        log.info(f"GCSP Sampling iter {i}")
        _ = env.reset()
        std = i / float(max_attempts) * max_noise
        noised_plan = add_gaussian_noise(ground_plan, std=std * env.param_scale)
        constraint_violated = False
        log.info(noised_plan)
        for ai, action in enumerate(noised_plan):
            _, _, _, info = env.step(action)
            if len(info["constraint_violations"]) > 0:
                violation_str = [
                    "Step {}, Action {}, Violation: {}".format(
                        ai, action.name, violation
                    )
                    for violation in info["constraint_violations"]
                ]
                violation_modes.update(violation_str)
                constraint_violated = True
                log.info(f"Constraint violation " + str(info["constraint_violations"]))
                break
        if not constraint_violated:
            return noised_plan, i

    return None, i


def parse_code(input_text):
    pattern = "```python(.*?)```"
    matches = re.findall(pattern, input_text, re.DOTALL)
    if len(matches) == 0:
        return None

    all_code = ""
    for match in matches:
        all_code += "\n" + match
    return all_code


def query_llm(messages, seed, max_retries=1):
    retry_count = 0
    backoff_factor = 60
    while True:
        try:
            st = time.time()
            completion = openai_client.chat.completions.create(
                model=ENGINE, messages=messages, seed=seed, temperature=0
            )

            return str(completion.choices[0].message.content), time.time() - st
        except openai.RateLimitError as e:
            retry_count += 1
            if retry_count > max_retries:
                raise e
            sleep_time = backoff_factor * (2**retry_count)
            print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

def query_llm_async(messages, seed, callback=None) -> Future:
    fut = _LLM_POOL.submit(query_llm, messages, seed)

    if callback is not None:
        def _on_done(f: Future):
            try:
                text, elapsed = f.result()
                callback(text, elapsed, None)
                print(text)
                print(callback)
            except Exception as err:
                callback(None, None, err)
                print(callback)
        fut.add_done_callback(_on_done)
    else:
        print("no callback")

    return fut

def get_prompt_element_map():

    folder_path = os.path.join(
        pathlib.Path(__file__).parent, "main_prompt_elements"
    )
    file_text_map = {}

    # Iterate over all the files in the specified folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                file_text_map[file_name.replace(".txt", "")] = text

    return file_text_map


def parse_text_prompt(text_prompt_path):
    entries = []
    current_role = None
    current_content = []
    prompt_elements = get_prompt_element_map()

    with open(text_prompt_path, "r") as file:
        template = file.read()
        #print(f"template {template}")
        rendered = pystache.render(template, prompt_elements)
        for line in rendered.split("\n"):
            if line.startswith("#define"):
                if current_role is not None:
                    # Join all content lines and strip to clean up whitespace
                    content = "".join(current_content)
                    entries.append({"role": current_role, "content": content})
                    current_content = []
                current_role = line.split()[-1]  # Get the last word, which is the role
            else:
                current_content.append(line + "\n")

        # Don't forget to add the last entry if there is one
        if current_role is not None and current_content:
            content = "".join(current_content)
            entries.append({"role": current_role, "content": content})

    return entries


# def write_prompt(path, entries):
#     with open(os.path.join(get_log_dir(), path), "w") as file:
#         for entry in entries:
#             # Write the role definition
#             file.write(f"#define {entry['role']}\n")
#             # Write the content, each line is separated
#             content_lines = entry["content"]

#             file.write(content_lines + "\n")






@dataclass
class Sampler:
    def sample(self):
        pass


@dataclass
class ContinuousSampler(Sampler):
    min: float = 0
    max: float = 1

    def sample(self):
        return random.uniform(self.min, self.max)


@dataclass
class DiscreteSampler:
    values: List[int]

    def sample(self):
        return random.choice(self.values)


class Policy(ABC):
    @abstractmethod
    def __init__(self, twin: Environment):
        self.twin = twin

    @abstractmethod
    def get_action(self, belief, goal: str, profile_stats={}):
        pass
