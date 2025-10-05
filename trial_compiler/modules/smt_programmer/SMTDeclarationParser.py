# modules/smt_declaration_parser.py

import re
import dspy
from typing import Dict, Any

class SMTDeclarationParser(dspy.Module):
    """
    Incrementally parses new SMT variable declarations and updates a
    dictionary mapping variable names to (type, description).
    """

    DECL_RE = re.compile(
        r"^\(declare-const\s+([^\s]+)\s+([^\s\)]+)\)\s*;?\s*(.*)$"
    )

    def __init__(self):
        super().__init__()
        self.variable_index: Dict[str, Dict[str, str]] = {}

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a new SMT slice, update the variable mapping.

        Expected input:
        - context["finalized_smt_slice"] : List[str] of SMT-LIB lines

        Output:
        - context["variable_index"] : cumulative var_name → { "type": ..., "description": ... }
        """

        lines = context.get("smt_program_lines", [])

        for line in lines:
            line = line.strip()
            match = self.DECL_RE.match(line)
            if match:
                var_name, var_type, comment = match.groups()
                if var_name not in self.variable_index:
                    self.variable_index[var_name] = {
                        "type": var_type,
                        "description": comment.strip()
                    }

        # Make it accessible to other modules
        context["variable_index"] = self.variable_index
        return context
