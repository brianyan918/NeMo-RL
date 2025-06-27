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

import re

from functools import partial

lang_lookup = {"en":"English", "cz": "Czech", "zh": "Chinese", "de": "German"}

def _extract_segment(text):
    """Extracts the single segment between <v> and </v> tags in the input string.

    Args:
        text (str): The input string containing one <v>...</v> segment.

    Returns:
        str: The extracted segment, or None if not found.
    """
    match = re.search(r'<v>(.*?)</v>', text)
    return match.group(1).strip() if match else None


def _cleanup(errors, span_type="none"):
    if span_type == "none":
        return [{"severity": x["severity"], "category": x["category"]} for x in errors]
    elif span_type == "tag":
        rv = []
        for x in errors:
            if "<v>" in x["src_span"] and "<v>" not in x["tgt_span"]:
                rv.append({"span": x["src_span"], "severity": x["severity"], "category": x["category"]})
            elif "<v>" not in x["src_span"] and "<v>" in x["tgt_span"]:
                rv.append({"span": x["tgt_span"], "severity": x["severity"], "category": x["category"]})
            elif "<v>" in x["src_span"] and "<v>" in x["tgt_span"]:
                rv.append({"src_span": x["src_span"], "tgt_span": x["tgt_span"], "severity": x["severity"], "category": x["category"]})
            else:
                rv.append({"severity": x["severity"], "category": x["category"]})
        return rv
    elif span_type == "seg":
        rv = []
        for x in errors:
            if "<v>" in x["src_span"] and "<v>" not in x["tgt_span"]:
                rv.append({"span": _extract_segment(x["src_span"]), "severity": x["severity"], "category": x["category"]})
            elif "<v>" not in x["src_span"] and "<v>" in x["tgt_span"]:
                rv.append({"span": _extract_segment(x["tgt_span"]), "severity": x["severity"], "category": x["category"]})
            elif "<v>" in x["src_span"] and "<v>" in x["tgt_span"]:
                rv.append({"src_span": _extract_segment(x["src_span"]), "tgt_span": _extract_segment(x["tgt_span"]), "severity": x["severity"], "category": x["category"]})
            else:
                rv.append({"severity": x["severity"], "category": x["category"]})
        return rv

def format_wmt(data: dict[str, Any], span_type="none") -> dict[str, list[dict[str, str]]]:
    src_lang=lang_lookup[data["lp"].split("-")[0]]
    src_text=data["src"]
    tgt_lang=lang_lookup[data["lp"].split("-")[-1]]
    tgt_text=data["tgt"]
    detailed_instruct_text = "\nBased on the source and target sentences surrounded with triple backticks ('''), identify error types in the translation and classify them. Please identify all errors within each translated segment, up to a maximum of five. If there are more than five errors, identify only the five most severe. The format of your output should be a json object in single line format. Directly generate this output without any additional reasoning.\nThe categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: major, minor, and neutral. Major errors inhibit comprehension of the text or disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension. No-errors should be marked as neutral.\n"
    user_content = f"{src_lang} source:\n'''{src_text}'''\n{tgt_lang} translation:\n'''{tgt_text}'''\n"
    errors = _cleanup(data["errors"], span_type)
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
    def __init__(self, span_type="none") -> None:
        self.span_type = span_type
        data_files = {
            "train": "../reasoning-mqm/data/train_data3.1.jsonl",
            "validation": "../reasoning-mqm/data/val_data3.1.jsonl"
        }
        original_ds = load_dataset("json", data_files=data_files)
        format_fxn = partial(format_wmt, span_type=self.span_type)
        self.formatted_ds = original_ds.map(format_fxn)
        self.task_spec = TaskDataSpec(
            task_name="WMT",
        )
