"""Tests for the search modules."""

import pytest

from core.search import fuse_results, get_model_display_name


class TestFuseResults:
    def test_fuse_results_basic(self):
        keyword = [
            {'file_path': 'a.cpp', 'start_line': 1, 'end_line': 10, 'score': 0.9, 'content': 'code a'},
            {'file_path': 'b.cpp', 'start_line': 5, 'end_line': 15, 'score': 0.7, 'content': 'code b'},
        ]
        unixcoder = [
            {'file_path': 'a.cpp', 'start_line': 1, 'end_line': 10, 'score': 0.8, 'content': 'code a'},
            {'file_path': 'c.cpp', 'start_line': 20, 'end_line': 30, 'score': 0.6, 'content': 'code c'},
        ]
        sbert = [
            {'file_path': 'a.cpp', 'start_line': 1, 'end_line': 10, 'score': 0.85, 'content': 'code a'},
        ]

        fused = fuse_results(keyword, unixcoder, sbert, max_results=5)

        assert len(fused) > 0
        # 'a.cpp:1-10' should be ranked first (appears in all 3 methods)
        assert fused[0]['rrf_score'] > fused[-1]['rrf_score']
        assert 'keyword' in fused[0]['source_methods']

    def test_fuse_results_empty(self):
        fused = fuse_results([], [], [])
        assert fused == []

    def test_fuse_results_single_method(self):
        keyword = [
            {'file_path': 'a.cpp', 'start_line': 1, 'end_line': 10, 'score': 0.9, 'content': 'code'},
        ]
        fused = fuse_results(keyword, [], [])
        assert len(fused) == 1
        assert fused[0]['source_methods'] == ['keyword']

    def test_fuse_results_max_results(self):
        results = [
            {'file_path': f'file{i}.cpp', 'start_line': 1, 'end_line': 10, 'score': 0.5, 'content': 'code'}
            for i in range(20)
        ]
        fused = fuse_results(results, [], [], max_results=5)
        assert len(fused) == 5


class TestDisplayNames:
    def test_known_types(self):
        assert get_model_display_name('keyword') == 'Keyword Search'
        assert get_model_display_name('unixcoder') == 'UniXcoder (Code Structure)'
        assert get_model_display_name('sbert') == 'SBERT (Semantic)'

    def test_unknown_type(self):
        assert get_model_display_name('unknown_model') == 'Unknown_Model'


class TestKeywordSearch:
    def test_extract_keywords(self):
        from core.keyword_search import ChromaDBKeywordSearch

        # Can't instantiate without ChromaDB, but we can test the static-like method
        engine = object.__new__(ChromaDBKeywordSearch)
        engine.verbose = False
        keywords = engine._extract_keywords("hello world test")
        assert keywords == ["hello", "world", "test"]

    def test_calculate_score(self):
        from core.keyword_search import ChromaDBKeywordSearch

        engine = object.__new__(ChromaDBKeywordSearch)
        engine.verbose = False

        score = engine._calculate_keyword_score("void Motor::step(double dt)", ["Motor", "step"])
        assert score > 0

        score_no_match = engine._calculate_keyword_score("void foo()", ["Motor"])
        assert score_no_match == 0.0

    def test_empty_keywords(self):
        from core.keyword_search import ChromaDBKeywordSearch

        engine = object.__new__(ChromaDBKeywordSearch)
        engine.verbose = False

        score = engine._calculate_keyword_score("some text", [])
        assert score == 0.0
