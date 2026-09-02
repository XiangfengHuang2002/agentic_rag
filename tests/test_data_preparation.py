import unittest

from src.data_preparation import DataPreparationPipeline, MAX_EMBEDDING_CHARS, clean_wiki_text, chunk_text


class TestDataPreparation(unittest.TestCase):
    def test_clean_wiki_text_removes_wiki_markup(self):
        raw = """
        <big>{{Color|#FF9900|'''请注意：不同区服的某些常用语存在细微差别'''}}</big><br>
        ==游戏常见用语==
        *FF14：最终幻想14的缩写。
        [[月读|月读（boss）]]
        [https://example.com/guide 说明链接]
        {{黑幕|隐藏文本}}
        """

        cleaned = clean_wiki_text(raw)

        self.assertIn("请注意", cleaned)
        self.assertIn("游戏常见用语", cleaned)
        self.assertNotIn("{{", cleaned)
        self.assertNotIn("[[", cleaned)
        self.assertNotIn("https://example.com", cleaned)
        self.assertIn("月读（boss）", cleaned)

    def test_chunk_text_splits_into_reasonable_segments(self):
        text = " ".join([
            "段落一。" * 18,
            "段落二。" * 18,
            "段落三。" * 18,
        ])

        chunks = chunk_text(text, chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.strip()) > 0 for chunk in chunks))
        self.assertTrue(all(len(chunk) <= MAX_EMBEDDING_CHARS for chunk in chunks))

    def test_chunk_text_limits_long_unbroken_input(self):
        source = "长文本" * 5000
        chunks = chunk_text(source, chunk_size=350, overlap=60)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= MAX_EMBEDDING_CHARS for chunk in chunks))

        covered = "".join(chunk if index == 0 else chunk[60:] for index, chunk in enumerate(chunks))
        self.assertEqual(covered, source)

    def test_chunk_text_keeps_long_sentence_tail(self):
        source = "开头。" + ("长句内容" * 500) + "。结尾有效信息。"
        chunks = chunk_text(source, chunk_size=350, overlap=60)

        self.assertIn("结尾有效信息。", "".join(chunk if index == 0 else chunk[60:] for index, chunk in enumerate(chunks)))

    def test_embedding_input_is_hard_limited(self):
        pipeline = object.__new__(DataPreparationPipeline)

        with self.assertRaises(ValueError):
            pipeline._embedding_for_text("x" * (MAX_EMBEDDING_CHARS + 1))


if __name__ == "__main__":
    unittest.main()
