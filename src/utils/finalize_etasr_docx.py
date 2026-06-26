"""Apply ETASR section layout to a Pandoc-generated revision DOCX."""

from __future__ import annotations

import argparse
import copy
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W, "wp": WP, "a": A, "m": M}


def qn(local_name: str) -> str:
    return f"{{{W}}}{local_name}"


def paragraph_text(paragraph) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def paragraph_properties(paragraph):
    properties = paragraph.find(qn("pPr"))
    if properties is None:
        properties = etree.Element(qn("pPr"))
        paragraph.insert(0, properties)
    return properties


def set_section_break(paragraph, section_properties, break_type: str) -> None:
    properties = paragraph_properties(paragraph)
    existing = properties.find(qn("sectPr"))
    if existing is not None:
        properties.remove(existing)
    section = copy.deepcopy(section_properties)
    section_type = section.find(qn("type"))
    if section_type is None:
        section_type = etree.Element(qn("type"))
        section.insert(0, section_type)
    section_type.set(qn("val"), break_type)
    properties.append(section)


def set_paragraph_alignment(paragraph, alignment: str) -> None:
    properties = paragraph_properties(paragraph)
    justification = properties.find(qn("jc"))
    if justification is None:
        justification = etree.Element(qn("jc"))
        properties.append(justification)
    justification.set(qn("val"), alignment)


def set_paragraph_style(paragraph, style_name: str) -> None:
    properties = paragraph_properties(paragraph)
    style = properties.find(qn("pStyle"))
    if style is None:
        style = etree.Element(qn("pStyle"))
        properties.insert(0, style)
    style.set(qn("val"), style_name)


def set_page_break_before(paragraph) -> None:
    properties = paragraph_properties(paragraph)
    if properties.find(qn("pageBreakBefore")) is None:
        properties.append(etree.Element(qn("pageBreakBefore")))


def add_right_tab_stop(paragraph, position_twips: int = 4320) -> None:
    properties = paragraph_properties(paragraph)
    tabs = properties.find(qn("tabs"))
    if tabs is None:
        tabs = etree.Element(qn("tabs"))
        properties.append(tabs)
    tab = etree.Element(qn("tab"))
    tab.set(qn("val"), "right")
    tab.set(qn("pos"), str(position_twips))
    tabs.append(tab)


def append_display_equation_numbers(
    paragraphs,
    start_index: int,
    end_index: int,
) -> int:
    number = 1
    for index, paragraph in enumerate(paragraphs):
        if index <= start_index or index >= end_index:
            continue
        if not paragraph.xpath(".//m:oMathPara", namespaces=NS):
            continue
        text = paragraph_text(paragraph).strip()
        if text.startswith("(") and text.endswith(")"):
            continue

        add_right_tab_stop(paragraph)
        number_run = etree.SubElement(paragraph, qn("r"))
        etree.SubElement(number_run, qn("tab"))
        number_text = etree.SubElement(number_run, qn("t"))
        number_text.text = f"({number})"
        number += 1
    return number - 1


def split_paragraph_at_first_break(paragraph):
    children = list(paragraph)
    break_child_index = None
    break_run = None
    for child_index, child in enumerate(children):
        if child.tag != qn("r"):
            continue
        if child.find(qn("br")) is not None:
            break_child_index = child_index
            break_run = child
            break
    if break_child_index is None:
        return paragraph, None

    second = etree.Element(qn("p"))
    for child in children[break_child_index + 1 :]:
        paragraph.remove(child)
        second.append(child)
    paragraph.remove(break_run)
    paragraph.addnext(second)
    return paragraph, second


def constrain_inline_images(root, maximum_width_emu: int = 2_925_000) -> None:
    for drawing in root.xpath(".//w:drawing", namespaces=NS):
        extent = drawing.find(".//wp:extent", namespaces=NS)
        transform_extent = drawing.find(".//a:xfrm/a:ext", namespaces=NS)
        if extent is None:
            continue
        width = int(extent.get("cx"))
        height = int(extent.get("cy"))
        if width <= maximum_width_emu:
            continue
        scale = maximum_width_emu / width
        new_height = int(height * scale)
        extent.set("cx", str(maximum_width_emu))
        extent.set("cy", str(new_height))
        if transform_extent is not None:
            transform_extent.set("cx", str(maximum_width_emu))
            transform_extent.set("cy", str(new_height))


def remove_manual_page_breaks(root) -> None:
    for page_break in root.xpath(
        './/w:br[@w:type="page"]',
        namespaces=NS,
    ):
        parent = page_break.getparent()
        parent.remove(page_break)


def load_package(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_docx", type=Path)
    parser.add_argument("template_docx", type=Path)
    parser.add_argument("output_docx", type=Path)
    arguments = parser.parse_args()

    package = load_package(arguments.input_docx)
    template = load_package(arguments.template_docx)

    document = etree.fromstring(package["word/document.xml"])
    template_document = etree.fromstring(template["word/document.xml"])
    template_sections = template_document.xpath(".//w:sectPr", namespaces=NS)
    if len(template_sections) < 2:
        raise RuntimeError("The ETASR template must contain one- and two-column sections")

    one_column = template_sections[0]
    two_column = template_sections[-1]
    paragraphs = document.xpath(".//w:p", namespaces=NS)

    manuscript_title = next(
        (
            paragraph
            for paragraph in paragraphs
            if paragraph_text(paragraph).strip().startswith(
                "Byzantine-Robust Federated Learning with Adaptive Aggregation"
            )
            and "Revised title:" not in paragraph_text(paragraph)
        ),
        None,
    )
    response_marker = next(
        (
            paragraph
            for paragraph in paragraphs
            if "REVISED MANUSCRIPT FOLLOWS" in paragraph_text(paragraph)
        ),
        None,
    )
    response_boundary = response_marker
    if response_boundary is None and manuscript_title is not None:
        title_index = paragraphs.index(manuscript_title)
        if title_index > 0:
            response_boundary = paragraphs[title_index - 1]
    keywords = next(
        (
            paragraph
            for paragraph in paragraphs
            if paragraph_text(paragraph).strip().startswith("Keywords:")
        ),
        None,
    )
    if response_boundary is None or keywords is None:
        raise RuntimeError("Could not locate response boundary or manuscript keywords")

    remove_manual_page_breaks(document)
    set_section_break(response_boundary, one_column, "nextPage")
    set_section_break(keywords, one_column, "continuous")
    if response_marker is not None:
        set_paragraph_alignment(response_marker, "center")

    response_heading = next(
        (
            paragraph
            for paragraph in paragraphs
            if paragraph_text(paragraph).strip()
            == "Response to Reviewer Comments"
        ),
        None,
    )
    abstract_heading = next(
        (
            paragraph
            for paragraph in paragraphs
            if paragraph_text(paragraph).strip() == "Abstract"
        ),
        None,
    )
    if response_heading is not None:
        set_paragraph_style(response_heading, "Title")
        set_paragraph_alignment(response_heading, "left")
    if manuscript_title is not None:
        set_paragraph_style(manuscript_title, "ETASRpapertitle")
        set_paragraph_alignment(manuscript_title, "left")
        set_page_break_before(manuscript_title)
    if abstract_heading is not None:
        set_paragraph_style(abstract_heading, "ETASRHeading5bold")
        set_paragraph_alignment(abstract_heading, "left")

    if manuscript_title is not None and abstract_heading is not None:
        start = paragraphs.index(manuscript_title)
        end = paragraphs.index(abstract_heading)
        author_paragraphs = paragraphs[start + 1 : end]
        for paragraph in author_paragraphs:
            author, affiliation = split_paragraph_at_first_break(paragraph)
            set_paragraph_style(author, "ETASRauthor")
            set_paragraph_alignment(author, "left")
            if affiliation is not None:
                set_paragraph_style(affiliation, "ETASRaffiliation")
                set_paragraph_alignment(affiliation, "left")

    paragraphs = document.xpath(".//w:p", namespaces=NS)
    manuscript_start = paragraphs.index(manuscript_title) if manuscript_title is not None else 0
    references_heading = next(
        (
            paragraph
            for paragraph in paragraphs
            if paragraph_text(paragraph).strip() == "References"
        ),
        None,
    )
    first_reference = next(
        (
            paragraph
            for paragraph in paragraphs
            if paragraph_text(paragraph).strip().startswith("H. B. McMahan")
        ),
        None,
    )
    if references_heading is None and first_reference is not None:
        first_reference_index = paragraphs.index(first_reference)
        if first_reference_index > 0:
            previous = paragraphs[first_reference_index - 1]
            if paragraph_text(previous).strip().isdigit():
                previous.getparent().remove(previous)
        references_heading = etree.Element(qn("p"))
        heading_run = etree.SubElement(references_heading, qn("r"))
        heading_text = etree.SubElement(heading_run, qn("t"))
        heading_text.text = "References"
        first_reference.addprevious(references_heading)
        set_paragraph_style(references_heading, "ETASRHeading5")

    paragraphs = document.xpath(".//w:p", namespaces=NS)
    references_start = (
        paragraphs.index(references_heading)
        if references_heading is not None
        else len(paragraphs)
    )

    for index, paragraph in enumerate(paragraphs):
        if index <= manuscript_start:
            continue
        text = paragraph_text(paragraph).strip()
        properties = paragraph.find(qn("pPr"))
        style = properties.find(qn("pStyle")) if properties is not None else None
        style_id = style.get(qn("val")) if style is not None else None

        if paragraph is abstract_heading:
            continue
        if text in {
            "Declarations",
            "Declaration of Competing Interests",
            "Acknowledgment",
            "Data Availability",
            "AI Use and Declaration of Generative AI Use",
            "References",
        }:
            set_paragraph_style(paragraph, "ETASRHeading5")
        elif text.startswith("Keywords:"):
            set_paragraph_style(paragraph, "ETASRkeywords")
            set_paragraph_alignment(paragraph, "both")
        elif index == paragraphs.index(abstract_heading) + 1:
            set_paragraph_style(paragraph, "ETASRabstract")
            set_paragraph_alignment(paragraph, "both")
        elif style_id == "Heading1":
            set_paragraph_style(paragraph, "ETASRHeading1")
        elif style_id == "Heading2":
            set_paragraph_style(paragraph, "ETASRHeading2")
        elif style_id == "Heading3":
            set_paragraph_style(paragraph, "ETASRHeading3")
        elif style_id == "TableCaption":
            set_paragraph_style(paragraph, "ETASRtablehead")
        elif style_id == "ImageCaption":
            set_paragraph_style(paragraph, "ETASRfigurecaption")
        elif index > references_start:
            set_paragraph_style(paragraph, "ETASRreferences")
            set_paragraph_alignment(paragraph, "both")
        elif paragraph.find(".//" + qn("numPr")) is not None or style_id is None:
            set_paragraph_style(paragraph, "ETASRbulletlist")
        elif style_id in {"BodyText", "FirstParagraph", "Compact"}:
            set_paragraph_style(paragraph, "ETASRbodytext")
            set_paragraph_alignment(paragraph, "both")

    for table in document.xpath(".//w:tbl", namespaces=NS):
        rows = table.xpath("./w:tr", namespaces=NS)
        for row_index, row in enumerate(rows):
            for paragraph in row.xpath(".//w:p", namespaces=NS):
                set_paragraph_style(
                    paragraph,
                    "ETASRtablecolhead" if row_index == 0 else "ETASRtablecopy",
                )

    numbered_equations = append_display_equation_numbers(
        paragraphs,
        manuscript_start,
        references_start,
    )
    constrain_inline_images(document)

    body = document.find(qn("body"))
    final_section = body.find(qn("sectPr"))
    if final_section is not None:
        body.remove(final_section)
    body.append(copy.deepcopy(two_column))

    package["word/document.xml"] = etree.tostring(
        document,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )

    for template_part in (
        "word/styles.xml",
        "word/theme/theme1.xml",
        "word/fontTable.xml",
    ):
        if template_part in template:
            package[template_part] = template[template_part]

    arguments.output_docx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        arguments.output_docx,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, data in package.items():
            archive.writestr(name, data)

    print(f"Finalized: {arguments.output_docx}")
    print(f"Numbered display equations: {numbered_equations}")


if __name__ == "__main__":
    main()
