#! /usr/bin/env python
# coding:utf-8


import unittest
import io
from emotion.classifier import DictionaryLoader
from emotion.classifier import PosNegClassifier


class TestDictionaryLoader(unittest.TestCase):
    def test_load(self):
        lines = ["p\t楽しい",
                 "n\tうつ 状態",
                 "p\t魅力 的 だ"]
        reader = io.StringIO("\n".join(lines))
        dictionary = DictionaryLoader().load(reader)
        self.assertEqual(dictionary.longest_word_len, 3)
        self.assertEqual(dictionary.word2label["楽しい"], "p")
        self.assertEqual(dictionary.word2label["うつ 状態"], "n")
        self.assertEqual(dictionary.word2label["魅力 的 だ"], "p")
        self.assertEqual(dictionary.label2words["p"], {"楽しい", "魅力 的 だ"})
        self.assertEqual(dictionary.label2words["n"], {"うつ 状態"})


class TestPosNegClassifier(unittest.TestCase):
    def test_predict(self):
        lines = ["p\t楽しい",
                 "n\tうつ 状態",
                 "p\t魅力 的 だ"]
        reader = io.StringIO("\n".join(lines))
        dictionary = DictionaryLoader().load(reader)
        classifier = PosNegClassifier(dictionary)
        cases = [("学校 が 楽しい", 1/3),
                 ("仕事 が 忙しい て うつ 状態 に なる た", -1/9),
                 ("楽しい 時 と うつ 状態 の とき が ある", 0)]
        for case in cases:
            text = case[0]
            ans = case[1]
            res = classifier.predict(text)
            self.assertAlmostEqual(res, ans)
