"""
cad/expression.py

Safe math expression evaluator for CAD input fields.

parse_expr(text, unit) -> float (mm)

Supports:
  - Arithmetic:  + - * / ** ( )
  - Constants:   pi  tau  e
  - Unit suffix: 10mm  2.5in  1ft  3cm  0.5m
                 (suffix overrides the *unit* argument for that term)
  - Active unit: bare numbers are interpreted in *unit* (default mm)

Examples:
  parse_expr("10 + 5/2", "mm")   -> 12.5
  parse_expr("1in + 5mm", "mm")  -> 30.4
  parse_expr("2 * pi * 3", "mm") -> 18.849...
  parse_expr("  50  ", "cm")     -> 500.0  (50 cm in mm)
"""

from __future__ import annotations
import ast
import math
import re
from cad.units import UNITS, to_mm

# ---------------------------------------------------------------------------
# Allowed names in expressions
# ---------------------------------------------------------------------------

_SAFE_NAMES: dict[str, float] = {
    "pi":  math.pi,
    "tau": math.tau,
    "e":   math.e,
}

# ---------------------------------------------------------------------------
# Pre-processing — replace unit suffixes before parsing
# ---------------------------------------------------------------------------

# Matches a number (int or float) immediately followed by a unit suffix.
# e.g. "10mm", "2.5in", "1ft", "3cm", "0.5m"
_UNIT_RE = re.compile(
    r'(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*(' +
    '|'.join(sorted(UNITS.keys(), key=len, reverse=True)) +  # longest first
    r')\b'
)


def _preprocess(text: str) -> str:
    """
    Replace unit suffixes with explicit multiplication by the unit's mm factor.

    e.g.  "10mm + 1in"  ->  "10*1.0 + 1*25.4"
          "2/3in"        ->  "2/3*25.4"   (correct: (2/3)*25.4 = 16.93mm)
          "2.5cm"        ->  "2.5*10.0"

    This preserves operator precedence: the unit factor multiplies only the
    adjacent number, and arithmetic around it evaluates left-to-right normally.
    """
    def replace(m: re.Match) -> str:
        value = m.group(1)
        unit  = m.group(2)
        factor = UNITS[unit]
        return f"{value}*{factor!r}"

    return _UNIT_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# AST evaluator — whitelist only safe node types
# ---------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)


def _eval_node(node: ast.AST) -> float:
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id not in _SAFE_NAMES:
            raise ValueError(f"Unknown name: {node.id!r}")
        return _SAFE_NAMES[node.id]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        op = node.op
        if isinstance(op, ast.Add):      return left + right
        if isinstance(op, ast.Sub):      return left - right
        if isinstance(op, ast.Mult):     return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        if isinstance(op, ast.FloorDiv):
            if right == 0:
                raise ValueError("Division by zero")
            return float(left // right)
        if isinstance(op, ast.Mod):      return left % right
        if isinstance(op, ast.Pow):      return left ** right
        raise ValueError("Unsupported binary operator")

    raise ValueError(f"Unhandled node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_expr(text: str, unit: str = "mm") -> float:
    """
    Parse and evaluate *text* as a math expression.

    Bare numbers are interpreted in *unit* and converted to mm.
    Unit-suffixed numbers (e.g. "1in") are always converted correctly
    regardless of *unit*.

    Returns the result in millimetres, or raises ValueError with a
    human-readable message on any error.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty expression")

    # Replace unit suffixes first, then scale bare numbers by active unit
    processed = _preprocess(text)

    # Wrap bare result in active-unit conversion: multiply by mm_per_unit
    # We do this by wrapping the whole expression: (expr) * factor
    # BUT only if no unit suffix was found in the original text — if the
    # user wrote "10mm" we already converted it; if they wrote "10" we scale.
    # Strategy: after preprocessing, any remaining plain number should be
    # scaled. We achieve this by wrapping: result = eval(processed) * factor
    # then checking if the original had any unit suffix to avoid double-scaling.
    has_explicit_unit = bool(_UNIT_RE.search(text))

    try:
        tree = ast.parse(processed, mode='eval')
    except SyntaxError as ex:
        raise ValueError(f"Syntax error: {ex.msg}") from ex

    try:
        result = _eval_node(tree)
    except (ValueError, ZeroDivisionError, OverflowError) as ex:
        raise ValueError(str(ex)) from ex

    # Scale by active unit only if the expression had no explicit unit suffixes
    # (mixed expressions like "10mm + 5" treat the bare 5 as mm, not active unit —
    # this is the least surprising behaviour)
    if not has_explicit_unit:
        from cad.units import UNITS
        result *= UNITS.get(unit, 1.0)

    return float(result)
