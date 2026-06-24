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
NS = {"w": W, "wp": WP, "a": A}


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

    response_marker = next(
        (
            paragraph
            for paragraph in paragraphs
            if "REVISED MANUSCRIPT FOLLOWS" in paragraph_text(paragraph)
        ),
        None,
    )
    keywords = next(
        (
            paragraph
            for paragraph in paragraphs
            if paragraph_text(paragraph).strip().startswith("Keywords:")
        ),
        None,
    )
    if response_marker is None or keywords is None:
        raise RuntimeError("Could not locate response marker or manuscript keywords")

    remove_manual_page_breaks(document)
    set_section_break(response_marker, one_column, "nextPage")
    set_section_break(keywords, one_column, "continuous")
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
        set_paragraph_style(manuscript_title, "Title")
        set_paragraph_alignment(manuscript_title, "center")
        set_page_break_before(manuscript_title)
    if abstract_heading is not None:
        set_paragraph_alignment(abstract_heading, "center")

    if manuscript_title is not None and abstract_heading is not None:
        start = paragraphs.index(manuscript_title)
        end = paragraphs.index(abstract_heading)
        for paragraph in paragraphs[start:end]:
            set_paragraph_alignment(paragraph, "center")

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


if __name__ == "__main__":
    main()
