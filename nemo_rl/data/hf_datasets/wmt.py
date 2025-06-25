# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any

from datasets import load_dataset, DatasetDict

from nemo_rl.data.interfaces import TaskDataSpec

import json

lang_lookup = {"en":"English", "cz": "Czech", "zh": "Chinese", "de": "German"}

def _cleanup(errors):
    return [{"severity": x["severity"], "category": x["category"]} for x in errors]

def format_wmt(data: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    src_lang=lang_lookup[data["lp"].split("-")[0]]
    src_text=data["src"]
    tgt_lang=lang_lookup[data["lp"].split("-")[-1]]
    tgt_text=data["tgt"]
    detailed_instruct_text = "\nBased on the source and target sentences surrounded with triple backticks ('''), identify error types in the translation and classify them. Please identify all errors within each translated segment, up to a maximum of five. If there are more than five errors, identify only the five most severe. The format of your output should be a json object in single line format. Directly generate this output without any additional reasoning.\nThe categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: major, minor, and neutral. Major errors inhibit comprehension of the text or disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension. No-errors should be marked as neutral.\n"
    user_content = f"{src_lang} source:\n'''{src_text}'''\n{tgt_lang} translation:\n'''{tgt_text}'''\n"
    errors = _cleanup(data["errors"])
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.",
            },
            {
                "role": "user",
                "content": user_content + detailed_instruct_text,
            },
            {
                "role": "assistant",
                "content": json.dumps({"errors": errors}),
            },
        ]
    }


class WMTDataset:
    def __init__(self) -> None:
        data_files = {
            "train": "../reasoning-mqm/data/pudding_train.jsonl",
            "validation": "../reasoning-mqm/data/pudding_dev.jsonl"
        }
        original_ds = load_dataset("json", data_files=data_files)
        self.formatted_ds = original_ds.map(format_wmt)
        self.task_spec = TaskDataSpec(
            task_name="WMT",
        )
