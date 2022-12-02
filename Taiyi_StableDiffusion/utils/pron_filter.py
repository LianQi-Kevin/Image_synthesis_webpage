import os
from string import punctuation as EN_punctuation
from string import whitespace
from zhon.hanzi import punctuation as ZH_punctuation


class blacklist_filter:
    def __init__(self):
        self.blacklist = []

    # read blacklist file
    def add_from_file(self, blacklist_path: str):
        assert os.path.exists(blacklist_path), "Blacklist file: {} not found".format(blacklist_path)
        with open(blacklist_path, "r") as f:
            self.blacklist = [word.replace('\n', '').replace('"', '') for word in f.readlines()]
            f.close()

    # add words
    def add_words(self, word_list: list):
        for word in word_list:
            self.blacklist.append(word)

    def censor(self, sentence: str) -> str:
        un_output = []
        word_list = self._replace_and_split(sentence)
        for word in word_list:
            if word in self.blacklist:
                un_output.append(word)
        for b_word in un_output:
            sentence = sentence.replace(b_word, "****")
        return sentence

    def is_clean(self, sentence: str) -> bool:
        if len(self._inter(self._replace_and_split(sentence), self.blacklist)) != 0:
            return False
        else:
            return True

    def is_profane(self, sentence: str) -> bool:
        return not self.is_clean(sentence)

    @staticmethod
    def _replace_and_split(sentence: str) -> list:
        for p in "".join([EN_punctuation, ZH_punctuation, whitespace]):
            sentence = sentence.replace(p, " ")
        return [word for word in sentence.split(" ") if word != ""]

    @staticmethod
    def _inter(a: list, b: list) -> list:
        return list(set(a) & set(b))


if __name__ == '__main__':
    words = "This is a test sentence"
    ProfanityFilter = blacklist_filter()
    ProfanityFilter.add_from_file("pron_blacklist.txt")

    print(ProfanityFilter.censor(words))
    print(ProfanityFilter.is_profane(words))