#! /usr/bin/env python
# coding:utf-8


class PosNegClassifier:
    def __init__(self, filename):
        """"""
        self._label_word_dict, \
            self._word_label_dict, \
            self._longest_word_len = self._load_dict(filename)

    def _load_dict(self, filename):
        label_word_dic = {"p": set(),
                          "n": set(),
                          "e": set()}
        word_label_dic = dict()
        longest_word_len = -1
        with open(filename) as f:
            for line in f:
                label, word = line.strip("\n").split("\t")
                label_word_dic[label].add(word)
                word_label_dic[word] = label
                len_word = len(word.split(" "))
                if len_word > longest_word_len:
                    longest_word_len = len_word
        return label_word_dic, word_label_dic, longest_word_len

    def predict(self, text):
        """
        Args:
            text (str): 原形で分かち書きされた文

        Returns:
            int: 次の計算式で表されるスコアを返す

                (ポジティブである単語数 - ネガティブである単語数) / 単語数
        """

        words = text.split(" ")
        len_words = len(words)
        score = 0
        for i in range(self._longest_word_len):
            lst = [words[j:j+i+1] for j in range(len_words - i)]
            for word in lst:
                st = " ".join(word)
                _label = self._word_label_dict.get(st, "e")
                if _label == "p":
                    score += 1
                elif _label == "n":
                    score -= 1
        return score / len(words)
