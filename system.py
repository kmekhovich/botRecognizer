import os
import psutil
from urllib.parse import quote
import requests


def printmemory(measure):
    process = psutil.Process(os.getpid())
    end = {
        0: 'b',
        1: 'Kb',
        2: 'Mb',
        3: 'Gb'
    }
    string = "Using RAM: {} {}".format(process.memory_info().rss / 2 ** (10 * measure), end[measure])
    return string


def manage_downloading(net):
    for breed in net.dataset.breeds:
        print(breed)
        term = "{}".format(breed)
        path = "Preview/{}.jpg".format(breed)
        download(term, path)


def download(searchTerm, path):
    query = "https://ru.wikipedia.org/wiki/{}".format(quote(searchTerm))
    print(query)
    searchUrl = requests.get(query).text
    url = "svg"
    ind = 1
    while "svg" in url:
        url = "http://{}".format(searchUrl.split('src=\"//')[ind].split("\"")[0])
        ind += 1
    print(url)
    with open(path, 'wb') as handle:
        response = requests.get(url, stream=True)
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
