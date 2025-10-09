import urllib.request
import os.path

the_verdict_path="../../data/the-verdict.txt"

def pull_the_verdict_text() -> None:
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = the_verdict_path
    urllib.request.urlretrieve(url, file_path)

def read_file_content(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text

if __name__ == "__main__":
    if not os.path.isfile(the_verdict_path):
        pull_the_verdict_text()

    read_file_content(the_verdict_path)
