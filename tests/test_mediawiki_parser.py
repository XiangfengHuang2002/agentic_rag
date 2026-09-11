from src.data_preparation import clean_wiki_text
from src.mediawiki_parser import parse_mediawiki


def test_clean_wiki_text_removes_markup():
    raw = "== 标题 ==\n[[链接]]\n{{模板|text=显示文本}}\n普通文本"

    cleaned = clean_wiki_text(raw)

    assert "标题" in cleaned
    assert "链接" in cleaned
    assert "显示文本" in cleaned
    assert "{{" not in cleaned and "[[" not in cleaned


def test_parse_mediawiki_extracts_headings_and_templates():
    raw = "== 章节A ==\n[[术语]]\n{{模板|text=模板显示值}}\n"

    document = parse_mediawiki(raw)

    assert any(item.title == "章节A" for item in document.headings)
    assert any(item.target == "术语" for item in document.links)
    assert "模板" in document.templates_by_name
    assert document.templates_by_name["模板"][0].display_text == "模板显示值"


def test_known_templates_preserve_semantic_names_and_types():
    raw = """{{xh|攻击|物理}} {{xh|深宵换装|无}} {{xl|月读|kk}}
{{状态|击倒|id=625}} {{Role|T|防护}}
{{对话|月读|第一句<br>第二句}} {{Color|#FF9900|注意}}"""

    document = parse_mediawiki(raw)

    assert "攻击(物理)" in document.plain_text
    assert "深宵换装" in document.plain_text
    assert "月读" in document.plain_text
    assert "月读(kk)" not in document.plain_text
    assert "击倒(625)" in document.plain_text
    assert "防护" in document.plain_text
    assert "月读：第一句\n第二句" in document.plain_text
    assert "注意" in document.plain_text


def test_skill_id_uses_local_action_mapping(monkeypatch):
    parser = __import__("src.mediawiki_parser", fromlist=["ACTION_ID_NAME"])
    monkeypatch.setattr(parser, "ACTION_ID_NAME", {"30": "无敌"})

    document = parse_mediawiki("{{技能|id=30|text}}")

    assert "无敌" in document.plain_text


def test_skill_id_falls_back_to_id_when_local_mapping_is_missing(monkeypatch):
    parser = __import__("src.mediawiki_parser", fromlist=["ACTION_ID_NAME"])
    monkeypatch.setattr(parser, "ACTION_ID_NAME", {})

    document = parse_mediawiki("{{技能|id=999999|text}}")

    assert "技能(id=999999)" in document.plain_text


def test_skill_state_achievement_and_media_nodes_keep_readable_content():
    raw = """{{技能|id=30|text}} {{技能|真北|text}} {{状态|id=700}}
{{成就|1364|暗龙克星}} [[文件:061523.png|链接=|32x32像素]]
<gallery>芝诺斯裙甲.png|芝诺斯裙甲</gallery><br>下一句"""

    document = parse_mediawiki(raw)

    assert "Hallowed Ground" in document.plain_text or "技能(id=30)" in document.plain_text
    assert "真北" in document.plain_text
    assert "状态(id=700)" in document.plain_text
    assert "暗龙克星" in document.plain_text
    assert "061523.png" not in document.plain_text
    assert "链接=" not in document.plain_text
    assert "芝诺斯裙甲" in document.plain_text
    assert "\n下一句" in document.plain_text