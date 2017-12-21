#! /usr/bin/env python
# coding:utf-8


import unittest
import os
from emotion.classifier import PosNegClassifier


class TestPosNegClassifier(unittest.TestCase):
    def setUp(self):
        dictpath = os.path.join(os.path.dirname(__file__),
                                "data/emotion.dict")
        self.classifier = PosNegClassifier(dictpath)

    def test_predict(self):
        cases = [("学校 が 楽しい", 1/3),
                 ("仕事 が 忙しい て うつ 状態 に なる た", -1/9),
                 ("楽しい 時 と うつ 状態 の とき が ある", 0)]
        for case in cases:
            text = case[0]
            ans = case[1]
            res = self.classifier.predict(text)
            self.assertAlmostEqual(res, ans)
