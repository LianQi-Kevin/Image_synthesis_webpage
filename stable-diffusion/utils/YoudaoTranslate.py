import requests
import hashlib
from uuid import uuid4
from json import loads
import time
from typing import Union


class YoudaoTranslate(object):
    def __init__(self, APP_KEY: str, APP_SECRET: str):
        # API_Message
        self.YOUDAO_URL = 'https://openapi.youdao.com/api'
        self.APP_KEY = APP_KEY
        self.APP_SECRET = APP_SECRET

        # Request
        self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    @staticmethod
    def _encrypt(signStr: str) -> str:
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(signStr.encode('utf-8'))
        return hash_algorithm.hexdigest()

    @staticmethod
    def _truncate(q: str) -> str:
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    def text_translate(self, text: str, source_language="auto", target_language="en") -> Union[None, str]:
        # if input text is None, not request API
        if text is None:
            return None

        # create request body
        curtime = str(int(time.time()))
        salt = str(uuid4())
        sign = self._encrypt(self.APP_KEY + self._truncate(text) + salt + curtime + self.APP_SECRET)
        data = {'from': source_language, 'to': target_language, 'signType': 'v3', 'curtime': curtime,
                'appKey': self.APP_KEY, 'q': text, 'salt': salt, 'sign': sign}

        # get response
        response = loads(requests.post(url=self.YOUDAO_URL, data=data, headers=self.headers).content)
        return response["translation"][0]


if __name__ == '__main__':
    # you can get this from https://ai.youdao.com/product-fanyi-text.s
    APP_KEY = 'YOUR APP KEY'
    APP_SECRET = 'YOUR APP SECRET'

    # the text which need to translate
    text = "The text will be translated"

    textTranslate = YoudaoTranslate(APP_KEY, APP_SECRET)
    print(textTranslate.text_translate(text=text, source_language="auto", target_language="zh-CHS"))
