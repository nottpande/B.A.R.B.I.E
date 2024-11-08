import requests
from bs4 import BeautifulSoup
import csv

def scrape_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find('div', id='mw-content-text')

    if content_div:
        paragraphs = content_div.find_all('p')
        article_text = '\n'.join([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])
        return article_text
    else:
        return None


article_pairs = [
    {
        'kannada_url': 'https://kn.wikipedia.org/wiki/%E0%B2%95%E0%B3%81%E0%B2%B5%E0%B3%86%E0%B2%82%E0%B2%AA%E0%B3%81',
        'english_url': 'https://en.wikipedia.org/wiki/Kuvempu'
    },
    {
        'kannada_url': 'https://kn.wikipedia.org/wiki/%E0%B2%B8%E0%B3%8D%E0%B2%B5%E0%B2%BE%E0%B2%AE%E0%B2%BF_%E0%B2%B5%E0%B2%BF%E0%B2%B5%E0%B3%87%E0%B2%95%E0%B2%BE%E0%B2%A8%E0%B2%82%E0%B2%A6',
        'english_url': 'https://en.wikipedia.org/wiki/Swami_Vivekananda'
    },
    {
        'kannada_url': 'https://kn.wikipedia.org/wiki/%E0%B2%A4%E0%B2%BE%E0%B2%9C%E0%B3%8D_%E0%B2%AE%E0%B2%B9%E0%B2%B2%E0%B3%8D',
        'english_url': 'https://en.wikipedia.org/wiki/Taj_Mahal'
    },
    {
        'kannada_url': 'https://kn.wikipedia.org/wiki/%E0%B2%B2%E0%B2%BF%E0%B2%AF%E0%B3%8A%E0%B2%A8%E0%B2%BE%E0%B2%B0%E0%B3%8D%E0%B2%A1%E0%B3%8A_%E0%B2%A1%E0%B2%BF%E0%B2%95%E0%B2%BE%E0%B2%AA%E0%B3%8D%E0%B2%B0%E0%B2%BF%E0%B2%AF%E0%B3%8A',
        'english_url': 'https://en.wikipedia.org/wiki/Leonardo_DiCaprio'
    }
]


with open('Data/kannada_english_parallel_corpus.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Kannada', 'English'])
    for pair in article_pairs:
        kannada_text = scrape_text(pair['kannada_url'])
        english_text = scrape_text(pair['english_url'])
        if kannada_text and english_text:
            writer.writerow([kannada_text, english_text])

print('Scraping and saving done.')
