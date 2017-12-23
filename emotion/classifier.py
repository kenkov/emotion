#! /usr/bin/env python
# coding:utf-8


class DictionaryLoader:
    def load_from_file(self, filename):
        with open(filename) as reader:
            return self.load(reader)

    def load(self, reader):
        label_word_dic = {"p": set(),
                          "n": set(),
                          "e": set()}
        word_label_dic = dict()
        longest_word_len = -1
        for line in reader:
            label, word = line.strip("\n").split("\t")
            label_word_dic[label].add(word)
            word_label_dic[word] = label
            len_word = len(word.split(" "))
            if len_word > longest_word_len:
                longest_word_len = len_word

        return Dictionary(word2label=word_label_dic,
                          label2words=label_word_dic)


class Dictionary:
    def __init__(self, word2label, label2words):
        self._w2l = word2label
        self._l2ws = label2words
        self._longest_word_len = max(len(word.split(" "))
                                     for word in self._w2l)

    @property
    def longest_word_len(self):
        return self._longest_word_len

    @property
    def word2label(self):
        return self._w2l

    @property
    def label2words(self):
        return self._l2ws


class PosNegClassifier:
    def __init__(self, dictionary):
        """"""
        self._dict = dictionary

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
        for i in range(self._dict.longest_word_len):
            lst = [words[j:j+i+1] for j in range(len_words - i)]
            for word in lst:
                st = " ".join(word)
                _label = self._dict.word2label.get(st, "e")
                if _label == "p":
                    score += 1
                elif _label == "n":
                    score -= 1
        return score / len(words)
