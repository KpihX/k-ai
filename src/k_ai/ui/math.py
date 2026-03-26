# src/k_ai/ui/math.py
"""
Zero-dependency LaTeX-to-Unicode renderer.

Converts LaTeX commands to Unicode equivalents (200+ symbols).
Handles Greek letters, operators, arrows, sets, fractions, superscripts,
subscripts, accents, and common math structures.

Ported from k_ui.render.scratch — self-contained, no external deps.
"""
import re
from typing import Optional

from rich.text import Text


# ===================================================================
# SYMBOL TABLE (200+ LaTeX → Unicode)
# ===================================================================

LATEX_SYMBOLS = {
    # ── Basic Operations & Relations ──────────────────────────
    "\\sqrt": "√", "\\pi": "π", "\\infty": "∞",
    "\\times": "×", "\\cdot": "·", "\\div": "÷",
    "\\pm": "±", "\\mp": "∓",
    "\\approx": "≈", "\\neq": "≠", "\\equiv": "≡",
    "\\leq": "≤", "\\geq": "≥",
    "\\ll": "≪", "\\gg": "≫",
    "\\propto": "∝", "\\sim": "∼",
    # ── Logic & Arrows ────────────────────────────────────────
    "\\rightarrow": "→", "\\leftarrow": "←",
    "\\Rightarrow": "⇒", "\\Leftarrow": "⇐",
    "\\leftrightarrow": "↔", "\\Leftrightarrow": "⇔",
    "\\implies": "⟹", "\\iff": "⟺",
    "\\mapsto": "↦", "\\to": "→",
    "\\uparrow": "↑", "\\downarrow": "↓",
    "\\hookrightarrow": "↪", "\\hookleftarrow": "↩",
    "\\longrightarrow": "⟶", "\\Longrightarrow": "⟹",
    "\\forall": "∀", "\\exists": "∃", "\\nexists": "∄",
    "\\therefore": "∴", "\\because": "∵",
    "\\neg": "¬", "\\lnot": "¬",
    "\\land": "∧", "\\lor": "∨",
    "\\vee": "∨", "\\wedge": "∧",
    # ── Set Theory ────────────────────────────────────────────
    "\\in": "∈", "\\notin": "∉",
    "\\subset": "⊂", "\\supset": "⊃",
    "\\subseteq": "⊆", "\\supseteq": "⊇",
    "\\cup": "∪", "\\cap": "∩",
    "\\emptyset": "∅", "\\varnothing": "∅",
    "\\setminus": "∖",
    # ── Calculus & Algebra ────────────────────────────────────
    "\\int": "∫", "\\iint": "∬", "\\iiint": "∭",
    "\\oint": "∮",
    "\\sum": "∑", "\\prod": "∏", "\\coprod": "∐",
    "\\partial": "∂", "\\nabla": "∇",
    "\\Delta": "Δ",
    # ── Greek Letters (Lowercase) ─────────────────────────────
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\epsilon": "ε", "\\varepsilon": "ε",
    "\\zeta": "ζ", "\\eta": "η", "\\theta": "θ", "\\vartheta": "ϑ",
    "\\iota": "ι", "\\kappa": "κ",
    "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ",
    "\\rho": "ρ", "\\sigma": "σ",
    "\\tau": "τ", "\\upsilon": "υ",
    "\\phi": "φ", "\\varphi": "ϕ",
    "\\chi": "χ", "\\psi": "ψ", "\\omega": "ω",
    # ── Greek Letters (Uppercase) ─────────────────────────────
    "\\Gamma": "Γ", "\\Theta": "Θ", "\\Lambda": "Λ",
    "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ",
    "\\Upsilon": "Υ", "\\Phi": "Φ", "\\Psi": "Ψ", "\\Omega": "Ω",
    # ── Accents (standalone) ──────────────────────────────────
    "\\hat": "^", "\\bar": "¯", "\\vec": "→",
    "\\dot": "·", "\\ddot": "¨", "\\tilde": "~",
    "\\text": "", "\\mathrm": "", "\\mathit": "", "\\mathbf": "",
    "\\operatorname": "",
    "\\left": "", "\\right": "",
    "\\big": "", "\\Big": "", "\\bigg": "", "\\Bigg": "",
    # ── Spacing ───────────────────────────────────────────────
    "\\,": " ", "\\;": " ", "\\:": " ", "\\!": "",
    "\\quad": "  ", "\\qquad": "    ", "~": " ",
    # ── Dots ──────────────────────────────────────────────────
    "\\cdots": "⋯", "\\ldots": "…", "\\vdots": "⋮", "\\ddots": "⋱", "\\dots": "…",
    # ── Brackets ──────────────────────────────────────────────
    "\\langle": "⟨", "\\rangle": "⟩",
    "\\lceil": "⌈", "\\rceil": "⌉",
    "\\lfloor": "⌊", "\\rfloor": "⌋",
    "\\lVert": "‖", "\\rVert": "‖", "\\|": "‖",
    # ── Common Functions ──────────────────────────────────────
    "\\sin": "sin", "\\cos": "cos", "\\tan": "tan",
    "\\arcsin": "arcsin", "\\arccos": "arccos", "\\arctan": "arctan",
    "\\sinh": "sinh", "\\cosh": "cosh", "\\tanh": "tanh",
    "\\log": "log", "\\ln": "ln", "\\exp": "exp",
    "\\lim": "lim", "\\sup": "sup", "\\inf": "inf",
    "\\min": "min", "\\max": "max", "\\det": "det",
    # ── Number Sets (mathbb) ──────────────────────────────────
    "\\mathbb{R}": "ℝ", "\\mathbb{N}": "ℕ", "\\mathbb{Z}": "ℤ",
    "\\mathbb{Q}": "ℚ", "\\mathbb{C}": "ℂ", "\\mathbb{P}": "ℙ",
    "\\mathbb{F}": "𝔽", "\\mathbb{H}": "ℍ",
    # ── Big Operators ─────────────────────────────────────────
    "\\bigoplus": "⨁", "\\bigotimes": "⨂",
    "\\bigcup": "⋃", "\\bigcap": "⋂",
    # ── Miscellaneous ─────────────────────────────────────────
    "\\star": "★", "\\circ": "∘", "\\bullet": "•",
    "\\angle": "∠", "\\perp": "⊥", "\\parallel": "∥",
    "\\cong": "≅", "\\simeq": "≃",
    "\\ell": "ℓ", "\\hbar": "ℏ",
    "\\Re": "ℜ", "\\Im": "ℑ", "\\aleph": "ℵ",
    "\\prime": "′",
    "\\oplus": "⊕", "\\otimes": "⊗",
    "\\vdash": "⊢", "\\dashv": "⊣", "\\models": "⊨",
    "\\top": "⊤", "\\bot": "⊥",
}

MATHCAL_MAP = {
    "A": "𝒜", "B": "ℬ", "C": "𝒞", "D": "𝒟", "E": "ℰ",
    "F": "ℱ", "G": "𝒢", "H": "ℋ", "I": "ℐ", "J": "𝒥",
    "K": "𝒦", "L": "ℒ", "M": "ℳ", "N": "𝒩", "O": "𝒪",
    "P": "𝒫", "Q": "𝒬", "R": "ℛ", "S": "𝒮", "T": "𝒯",
    "U": "𝒰", "V": "𝒱", "W": "𝒲", "X": "𝒳", "Y": "𝒴", "Z": "𝒵",
}

SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "n": "ⁿ", "i": "ⁱ", "x": "ˣ",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
    "k": "ᵏ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ", "t": "ᵗ",
}

SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "i": "ᵢ", "j": "ⱼ",
    "a": "ₐ", "e": "ₑ", "o": "ₒ", "x": "ₓ",
    "k": "ₖ", "n": "ₙ", "p": "ₚ", "r": "ᵣ",
}

# Regex for detecting math blocks in text
MATH_REGEX = re.compile(
    r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|"
    r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    re.DOTALL,
)


def strip_delimiters(latex: str) -> str:
    """Remove LaTeX math delimiters ($$, $, \\[, \\()."""
    latex = latex.strip()
    for start, end in [("$$", "$$"), ("\\[", "\\]"), ("$", "$"), ("\\(", "\\)")]:
        if latex.startswith(start) and latex.endswith(end):
            return latex[len(start):-len(end)].strip()
    return latex


def _is_simple(s: str) -> bool:
    if not s or len(s) == 1 or s.isdigit():
        return True
    if s.startswith("\\") and " " not in s and "{" not in s:
        return True
    return False


def latex_to_unicode(latex: str) -> str:
    """Convert a LaTeX math expression to Unicode text."""
    text = strip_delimiters(latex)

    # Mathcal
    for char, cal in MATHCAL_MAP.items():
        text = text.replace(f"\\mathcal{{{char}}}", cal)

    # Accents: \vec{x} → x⃗
    accent_map = {
        "vec": "\u20d7", "hat": "\u0302", "bar": "\u0304",
        "dot": "\u0307", "ddot": "\u0308", "tilde": "\u0303",
    }
    for cmd, combining in accent_map.items():
        pat = re.compile(fr'\\{cmd}\s*\{{([^{{}}]*)\}}')
        while True:
            new = pat.sub(fr'\1{combining}', text)
            if new == text:
                break
            text = new

    # \sqrt{x} → √(x)
    sqrt_pat = re.compile(r'\\sqrt\s*\{([^{}]*)\}')
    while True:
        new = sqrt_pat.sub(lambda m: f"√{m.group(1)}" if _is_simple(m.group(1).strip()) else f"√({m.group(1).strip()})", text)
        if new == text:
            break
        text = new

    # \frac{a}{b} → a/b or (a)/(b)
    frac_pat = re.compile(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}')
    while True:
        def _frac(m):
            n, d = m.group(1).strip(), m.group(2).strip()
            ns = n if _is_simple(n) else f"({n})"
            ds = d if _is_simple(d) else f"({d})"
            return f"{ns}/{ds}"
        new = frac_pat.sub(_frac, text)
        if new == text:
            break
        text = new

    # \binom{n}{k} → C(n,k)
    binom_pat = re.compile(r'\\binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}')
    while True:
        new = binom_pat.sub(r'C(\1,\2)', text)
        if new == text:
            break
        text = new

    # \boxed{x} → | x |
    boxed_pat = re.compile(r'\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')
    while True:
        new = boxed_pat.sub(r'| \1 |', text)
        if new == text:
            break
        text = new

    # Symbol replacements (longest first)
    for cmd, repl in sorted(LATEX_SYMBOLS.items(), key=lambda x: -len(x[0])):
        text = text.replace(cmd, repl)

    # Superscripts: ^{abc} and ^x
    def _sup_group(m):
        return "".join(SUPERSCRIPTS.get(c, c) for c in m.group(1))
    text = re.sub(r'\^\{([^{}]+)\}', _sup_group, text)
    for ch, sup in SUPERSCRIPTS.items():
        text = text.replace(f"^{ch}", sup)

    # Subscripts: _{abc} and _x
    def _sub_group(m):
        return "".join(SUBSCRIPTS.get(c, c) for c in m.group(1))
    text = re.sub(r'_\{([^{}]+)\}', _sub_group, text)
    for ch, sub in SUBSCRIPTS.items():
        text = text.replace(f"_{ch}", sub)

    # Cleanup
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_block_math(segment: str) -> bool:
    """True if segment is block math ($$, \\[)."""
    return segment.startswith("$$") or segment.startswith("\\[")


def render_math_text(text: str) -> Text:
    """Render a Rich Text with math segments converted to Unicode."""
    return Text(latex_to_unicode(text))
