#! /usr/bin/env python
# coding:utf-8


import os
from emotion.classifier import PosNegClassifier


def test_PosNegClassifier():
    dictpath = os.path.join(os.path.dirname(__file__),
                            "data/emotion.dict")
    classifier = PosNegClassifier(dictpath)

    text = "学校 が 楽しい"
    res = classifier.predict(text)
    assert res == 1 / 3

    # 元文「仕事が忙しくてうつ状態になった」
    text = "仕事 が 忙しい て うつ 状態 に なる た"
    res = classifier.predict(text)
    assert res == -1 / 9

    # 元文「楽しくて魅力的だ」
    text = "楽しい て 魅力 的 だ"
    res = classifier.predict(text)
    assert res == 2 / 5

    # 元文「楽しい時とうつ状態のときがある」
    text = "楽しい 時 と うつ 状態 の とき が ある"
    res = classifier.predict(text)
    assert res == 0