import html
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import mwparserfromhell
from mwparserfromhell.nodes import Comment, ExternalLink, Heading, Tag, Template, Text, Wikilink
from src.action_mapping import merge_action_mapping

ACTION_ID_NAME = merge_action_mapping()


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
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n[ \t]*", "\n", value)
    return value.strip()


def _template_record(node: Template) -> TemplateRecord:
    name = _clean_display_text(str(node.name)).strip()
    parameters: Dict[str, str] = {}
    positional: List[str] = []
    for parameter in node.params:
        key = _clean_display_text(str(parameter.name))
        value = _clean_display_text(_node_text(mwparserfromhell.parse(str(parameter.value))))
        if parameter.showkey:
            parameters[key] = value
        else:
            positional.append(value)

    normalized_name = name.casefold()
    display_modes = {"text", "list", "icon", "link", "plain", "raw"}
    meaningful = [value for value in positional if value.casefold() not in display_modes]
    display_text = ""
    if normalized_name in {"xh", "技能伤害"}:
        display_text = meaningful[0] if meaningful else ""
        damage_type = meaningful[1] if len(meaningful) > 1 else ""
        if damage_type and damage_type != "无":
            display_text = f"{display_text}({damage_type})"
    elif normalized_name == "xl":
        display_text = meaningful[0] if meaningful else ""
    elif name in {"状态", "技能", "物品", "副本", "任务", "地图", "版本", "成就", "Fate", "天气", "货币"}:
        action_id = parameters.get("id", "")
        if name == "技能" and action_id and not meaningful:
            display_text = ACTION_ID_NAME.get(action_id, f"技能(id={action_id})")
        elif name == "成就" and len(meaningful) > 1:
            display_text = meaningful[1]
        else:
            display_text = meaningful[0] if meaningful else ""
        if name == "状态" and parameters.get("id") and display_text:
            display_text = f"{display_text}({parameters['id']})"
        elif not display_text and action_id:
            display_text = f"{name}(id={action_id})"
        elif not display_text:
            display_text = name
    elif normalized_name == "role":
        display_text = meaningful[-1] if meaningful else parameters.get("text", "")
    elif name == "对话":
        if len(meaningful) >= 2:
            display_text = f"{meaningful[0]}：{meaningful[1]}"
        elif meaningful:
            display_text = meaningful[0]
    elif normalized_name in {"color", "黑幕", "需要长期更新"}:
        display_text = meaningful[-1] if meaningful else parameters.get("text", "")
    else:
        preferred_keys = ("text", "title", "name", "名称", "显示", "display", "label")
        display_text = next((parameters[key] for key in preferred_keys if parameters.get(key)), "")
        if not display_text:
            display_text = meaningful[0] if meaningful else name

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
    target_prefix = target.split(":", 1)[0].casefold()
    is_media_link = (
        target_prefix in {"file", "文件", "image"}
        or bool(re.match(r"^(?:file|image|文件)\s*:", target, re.I))
        or bool(re.search(r"\.(?:png|jpe?g|gif|webp|svg)(?:$|\?)", target, re.I))
    )
    if is_media_link:
        parts = [part.strip() for part in label.split("|")]
        captions = [part for part in parts if part and "=" not in part and not re.fullmatch(r"\d+(?:x\d+)?(?:像素|px)?", part, re.I)]
        label = captions[-1] if captions else ""
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
            link = _link_record(node)
            if _is_media_target(str(node.title)):
                output.append(_media_caption(str(node.text) if node.text is not None else ""))
            else:
                output.append(link.label)
        elif isinstance(node, ExternalLink):
            output.append(_external_link_record(node).label)
        elif isinstance(node, Heading):
            output.append(_node_text(node.title))
        elif isinstance(node, Tag):
            if node.tag.lower() == "br":
                output.append("\n")
                continue
            if node.tag.lower() == "gallery":
                output.append(_gallery_text(node.contents))
                continue
            if node.contents is not None:
                output.append(_node_text(node.contents))
        else:
            output.append(str(node))
    return "".join(output)


def _gallery_text(contents: Any) -> str:
    if contents is None:
        return ""
    lines = []
    for line in str(contents).splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) > 1 and parts[-1]:
            lines.append(parts[-1])
    return "\n".join(lines)


def _is_media_target(target: str) -> bool:
    return bool(re.search(r"(?:^|:)(?:file|image|文件)\s*:", target, re.I) or re.search(r"\.(?:png|jpe?g|gif|webp|svg)(?:$|\?)", target, re.I))


def _media_caption(text: str) -> str:
    parts = [part.strip() for part in text.split("|")]
    captions = [
        part for part in parts
        if part
        and "=" not in part
        and not re.fullmatch(r"\d+(?:x\d+)?(?:像素|px)?", part, re.I)
    ]
    return captions[-1] if captions else ""


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
