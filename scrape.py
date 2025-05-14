import requests
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

journals = [
    'https://jurnal.ugm.ac.id/ijccs/article/view/72075',
    'https://jurnal.ugm.ac.id/jmdt/article/view/85645',
    'https://jurnal.ugm.ac.id/ijccs/article/view/94843'
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

abstracts = []

def create_csv():
    with open("cleaned_data.csv", "w") as f:
        f.write("abstract,source\n") 
        f.close()

def scrape_journals():
    for url in journals:
        print(f"Scraping: {url}")
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to retrieve: {url}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        abstract_section = soup.find(id="articleAbstract")

        if not abstract_section:
            print("No abstract found.")
            continue

        first_div = abstract_section.find('div')
        if first_div:
            raw_text = first_div.get_text(strip=True)
            abstracts.append(raw_text)
        else:
            print("No <div> inside abstract section.\n")

def remove_stopwords():
    for idx, text in enumerate(abstracts):
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        filtered = [word for word in tokens if word not in stop_words]
        print(f"Abstract {idx + 1}: {' '.join(filtered)}\n")

def write_csv():
    with open ("cleaned_data.csv", "a") as f:
        for idx, _ in enumerate(abstracts):
            f.write(f'{abstracts[idx]},{journals[idx]}\n')
            f.close()

if __name__ == "__main__":
    create_csv()
    scrape_journals()
    remove_stopwords()
    write_csv()
