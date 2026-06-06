import urllib.request
from bs4 import BeautifulSoup

url = "https://www.gutenberg.org/cache/epub/1260/pg1260-images.html"
with urllib.request.urlopen(url) as response:
    html = response.read()

soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text()

start_marker = "***START OF THE PROJECT GUTENBERG EBOOK"
end_marker = "***END OF THE PROJECT GUTENBERG EBOOK"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx != -1 and end_idx != -1:
    start_idx = text.find('\n', start_idx) + 1
    book_text = text[start_idx:end_idx].strip()
    print(len(book_text))
