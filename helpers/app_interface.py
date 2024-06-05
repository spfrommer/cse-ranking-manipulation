import time
from typing import List
import random
import string
import bs4
import requests
from requests.auth import HTTPBasicAuth
import os
import numpy as np

from bs4 import BeautifulSoup

SERVER_URL = 'https://www.consumerproduct.org'

def upload_htmls_and_get_urls(htmls: List[str]) -> List[str]:
    response = requests.post(
        f'{SERVER_URL}/clear',
        auth=HTTPBasicAuth('admin', '***ENTER PASSWORD***')
    )
    if response.status_code != 200:
        raise Exception(f'Error clearing: {response.status_code}, {response.text}')

    urls = []
    for html in htmls:
        filename = ''.join(random.choices(string.ascii_letters, k=10)) + '.html'

        response = requests.put(
            f'{SERVER_URL}/upload/{filename}',
            data=html,
            auth=HTTPBasicAuth('admin', 'samreallysecretpwd155')
        )

        if response.status_code == 201:
            urls.append(f'{SERVER_URL}/{filename}')
        else:
            raise Exception(
                f'Error uploading {filename}: {response.status_code}, {response.text}'
            )

    # Try getting urls every second, only return when all succeed
    successfully_retrieved = [False for _ in urls]
    timeout = time.time() + 30
    while not all(successfully_retrieved):
        for i, url in enumerate(urls):
            if successfully_retrieved[i]:
                continue
            response = requests.get(url)
            if response.status_code == 200:
                successfully_retrieved[i] = True

        time.sleep(0.2)

        if time.time() > timeout:
            raise Exception('Timeout waiting for all files to be available')

    return urls

def poison_html(html: str, prompt: str, intersperse_n: int = 15) -> str:
    # Intersperse prompt into the text content of the HTML
    soup = BeautifulSoup(html, 'html.parser')

    all_text = soup.get_text()

    intersperse_locs = np.linspace(0, len(all_text), intersperse_n).astype(int)

    cum_text = ''

    replacements = []

    for t in soup.recursiveChildGenerator():
        if not isinstance(t, bs4.element.NavigableString):
            continue

        init_text = t.get_text()

        if len(init_text) == 0:
            continue

        cum_text += init_text

        locs_within_text_length = intersperse_locs[intersperse_locs <= len(init_text)]

        if len(locs_within_text_length) > 0:
            text = init_text
            to_cat = []
            for loc in locs_within_text_length:
                to_cat.append(text[:loc])
                to_cat.append(' ' + prompt + ' ')
                text = text[loc:]
            to_cat.append(text)
            resulting_text = ''.join(to_cat)

            replacements.append((t, resulting_text))

        intersperse_locs -= len(init_text)
        intersperse_locs = intersperse_locs[intersperse_locs > 0]

    for t, replacement in replacements:
        t.replace_with(replacement)

    return str(soup)
