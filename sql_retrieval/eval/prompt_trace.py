'''Used to trace the number of input/output tokens'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Optional

try:
    import tiktoken
except Exception:
    tiktoken = None


@dataclass
class TokenTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class PromptTrace:
    """
    Thread-safe prompt+output dumper + token counter.
    Appends to a single text file.
    """
    def __init__(self, dump_path: Path, tokenizer_model: str = "gpt-4o"):
        self.dump_path = dump_path
        self.lock = threading.Lock()
        self.totals = TokenTotals()

        self._enc = None
        if tiktoken is not None:
            try:
                self._enc = tiktoken.encoding_for_model(tokenizer_model)
            except Exception:
                self._enc = tiktoken.get_encoding("o200k_base")

    def _count_tokens(self, text: str) -> int:
        if self._enc is None:
            return 0
        return len(self._enc.encode(text))

    def log(self, *, stage: str, patient_id: str, trial_id: str, prompt: str, output: str) -> None:
        p_tok = self._count_tokens(prompt)
        o_tok = self._count_tokens(output)

        with self.lock:
            self.totals.prompt_tokens += p_tok
            self.totals.completion_tokens += o_tok

            with self.dump_path.open("a", encoding="utf-8") as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"PAIR: patient_id={patient_id} trial_id={trial_id}\n")
                f.write(f"STAGE: {stage}\n")
                f.write(f"TOKENS: prompt={p_tok} output={o_tok} (running_total={self.totals.total})\n")
                f.write("=" * 100 + "\n\n")

                f.write("----- PROMPT BEGIN -----\n")
                f.write(prompt)
                if not prompt.endswith("\n"):
                    f.write("\n")
                f.write("----- PROMPT END -----\n\n")

                f.write("----- OUTPUT BEGIN -----\n")
                f.write(output)
                if not output.endswith("\n"):
                    f.write("\n")
                f.write("----- OUTPUT END -----\n")
