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
