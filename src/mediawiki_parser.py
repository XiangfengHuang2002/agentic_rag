import html
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import mwparserfromhell
from mwparserfromhell.nodes import Comment, ExternalLink, Heading, Tag, Template, Text, Wikilink


@dataclass
class TemplateRecord:
    name: str
    parameters: Dict[str, str] = field(default_factory=dict)
    positional_parameters: List[str] = field(default_factory=list)
    raw: str = ""
    display_text: str = ""


@dataclass
class LinkRecord:
    target: str
    label: str
    namespace: str = ""
    section: str = ""
    raw: str = ""


@dataclass
class HeadingRecord:
    level: int
    title: str
    position: int = 0


@dataclass
class SectionRecord:
    level: int
    title: str
    text: str
    templates: List[str] = field(default_factory=list)


@dataclass
class MediaWikiDocument:
    plain_text: str
    headings: List[HeadingRecord] = field(default_factory=list)
    sections: List[SectionRecord] = field(default_factory=list)
    links: List[LinkRecord] = field(default_factory=list)
    templates: List[TemplateRecord] = field(default_factory=list)
    templates_by_name: Dict[str, List[TemplateRecord]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_display_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"'{2,5}", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _template_record(node: Template) -> TemplateRecord:
    name = _clean_display_text(str(node.name)).strip()
    parameters: Dict[str, str] = {}
    positional: List[str] = []
    for parameter in node.params:
        key = _clean_display_text(str(parameter.name))
        value = _clean_display_text(str(parameter.value))
        if parameter.showkey:
            parameters[key] = value
        else:
            positional.append(value)

    preferred_keys = ("text", "title", "name", "名称", "显示", "display", "label")
    display_text = next((parameters[key] for key in preferred_keys if parameters.get(key)), "")
    if not display_text:
        display_text = next((value for value in reversed(positional) if value), "")
    if not display_text and name.lower() not in {"color", "黑幕", "需要长期更新"}:
        display_text = name

    return TemplateRecord(
        name=name,
        parameters=parameters,
        positional_parameters=positional,
        raw=str(node),
        display_text=display_text,
    )


def _link_record(node: Wikilink) -> LinkRecord:
    target = _clean_display_text(str(node.title))
    label = _clean_display_text(str(node.text)) if node.text is not None else target
    namespace = target.split(":", 1)[0] if ":" in target else ""
    section = target.split("#", 1)[1] if "#" in target else ""
    return LinkRecord(target=target, label=label or target, namespace=namespace, section=section, raw=str(node))


def _external_link_record(node: ExternalLink) -> LinkRecord:
    target = _clean_display_text(str(node.url))
    label = _clean_display_text(str(node.title)) if node.title is not None else target
    return LinkRecord(target=target, label=label or target, raw=str(node))


def _node_text(code: mwparserfromhell.wikicode.Wikicode) -> str:
    """Render nodes to readable text while preserving useful template values."""
    output: List[str] = []
    for node in code.nodes:
        if isinstance(node, Comment):
            continue
        if isinstance(node, Text):
            output.append(str(node))
        elif isinstance(node, Template):
            record = _template_record(node)
            if record.display_text:
                output.append(record.display_text)
        elif isinstance(node, Wikilink):
            output.append(_link_record(node).label)
        elif isinstance(node, ExternalLink):
            output.append(_external_link_record(node).label)
        elif isinstance(node, Heading):
            output.append(_node_text(node.title))
        elif isinstance(node, Tag):
            if node.contents is not None:
                output.append(_node_text(node.contents))
        else:
            output.append(str(node))
    return "".join(output)


def _normalize_plain_text(text: str) -> str:
    text = html.unescape(text).replace("\r", "\n")
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?，。！？])", r"\1", text)
    return text.strip()


def parse_mediawiki(raw_text: str) -> MediaWikiDocument:
    """Parse MediaWiki markup into headings, sections, links and templates."""
    if raw_text is None:
        raw_text = ""
    wikicode = mwparserfromhell.parse(str(raw_text))

    headings: List[HeadingRecord] = []
    links: List[LinkRecord] = []
    templates: List[TemplateRecord] = []
    positions: Dict[int, int] = defaultdict(int)

    for node in wikicode.nodes:
        if isinstance(node, Heading):
            title = _normalize_plain_text(_node_text(node.title))
            headings.append(HeadingRecord(level=node.level, title=title, position=positions[node.level]))
            positions[node.level] += 1
    # 顶层过滤会递归捕获普通文本节点中的嵌套模板和链接。
    templates = [_template_record(node) for node in wikicode.ifilter_templates(recursive=True)]
    links = [_link_record(node) for node in wikicode.ifilter_wikilinks(recursive=True)]
    links.extend(_external_link_record(node) for node in wikicode.ifilter_external_links(recursive=True))

    templates_by_name: Dict[str, List[TemplateRecord]] = defaultdict(list)
    for template in templates:
        templates_by_name[template.name].append(template)

    plain_text = _normalize_plain_text(_node_text(wikicode))
    sections: List[SectionRecord] = []
    section_blocks = re.split(r"(?m)(?=^={2,6}[^=].*?={2,6}\s*$)", str(raw_text))
    for block in section_blocks:
        if not block.strip():
            continue
        match = re.match(r"(?m)^(={2,6})([^=].*?)\1\s*$", block)
        if match:
            level = len(match.group(1))
            title = _normalize_plain_text(match.group(2))
            body = block[match.end():]
        else:
            level = 0
            title = ""
            body = block
        section_code = mwparserfromhell.parse(body)
        section_text = _normalize_plain_text(_node_text(section_code))
        section_templates = [_clean_display_text(str(node.name)) for node in section_code.ifilter_templates(recursive=True)]
        sections.append(SectionRecord(level=level, title=title, text=section_text, templates=section_templates))

    return MediaWikiDocument(
        plain_text=plain_text,
        headings=headings,
        sections=sections,
        links=links,
        templates=templates,
        templates_by_name=dict(templates_by_name),
    )


def extract_templates_by_name(raw_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience API for callers that only need template classification."""
    document = parse_mediawiki(raw_text)
    return {
        name: [asdict(template) for template in records]
        for name, records in document.templates_by_name.items()
    }
