"""Prepare the ETASR LaTeX source for lossless Pandoc DOCX conversion."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_citations(source: str) -> str:
    bibliography_keys = re.findall(r"\\bibitem\{([^}]+)\}", source)
    citation_numbers = {
        key: index for index, key in enumerate(bibliography_keys, start=1)
    }

    def replace(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",")]
        numbers = [citation_numbers[key] for key in keys]
        return "[" + ", ".join(str(number) for number in numbers) + "]"

    return re.sub(r"\\cite\{([^}]+)\}", replace, source)


def replace_paths(source: str) -> str:
    def replace(match: re.Match[str]) -> str:
        escaped = match.group(1).replace("_", r"\_")
        return rf"\texttt{{{escaped}}}"

    return re.sub(r"\\path\{([^}]+)\}", replace, source)


def prepare(source: str) -> str:
    source = replace_citations(source)
    source = replace_paths(source)
    source = re.sub(r"(?m)^\\twocolumn\[\r?\n", "", source)
    source = re.sub(
        r"(?m)^\\begin\{minipage\}\{\\textwidth\}\r?\n",
        "",
        source,
    )
    source = re.sub(r"(?m)^\\raggedright\r?\n", "", source)
    source = re.sub(r"(?m)^\\end\{minipage\}\r?\n", "", source)
    source = re.sub(r"(?m)^\]\r?\n", "", source)
    source = source.replace(r"\begin{abstract}", r"\section*{Abstract}")
    source = source.replace(r"\end{abstract}", "")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tex", type=Path)
    parser.add_argument("output_tex", type=Path)
    arguments = parser.parse_args()

    source = arguments.input_tex.read_text(encoding="utf-8")
    arguments.output_tex.write_text(prepare(source), encoding="utf-8")
    print(f"Prepared: {arguments.output_tex}")


if __name__ == "__main__":
    main()
