import requests
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
cleaned_abstracts = []

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
    global cleaned_abstracts
    cleaned_abstracts = []
    for idx, text in enumerate(abstracts):
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        filtered = [word for word in tokens if word not in stop_words]
        cleaned = ' '.join(filtered)
        cleaned_abstracts.append(cleaned)
        print(f"Abstract {idx + 1} (cleaned): {cleaned}\n")

def write_csv():
    with open("cleaned_data.csv", "a") as f:
        for idx in range(len(abstracts)):
            f.write(f'"{abstracts[idx]}","{journals[idx]}"\n')

def print_top_keywords(vectorizer, tfidf_matrix):
    feature_names = vectorizer.get_feature_names_out()
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-5:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        print(f"Top keywords for Document {i+1}: {', '.join(top_terms)}\n")


def compute_similarity():
    print("\nComputing TF-IDF and Cosine Similarity...\n")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_abstracts)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print_top_keywords(vectorizer, tfidf_matrix)


    print("Cosine Similarity Matrix:")
    for i in range(len(cosine_sim)):
        for j in range(len(cosine_sim[i])):
            print(f"Doc {i+1} vs Doc {j+1}: {cosine_sim[i][j]:.4f}")
        print()


if __name__ == "__main__":
    create_csv()
    scrape_journals()
    remove_stopwords()
    write_csv()
    compute_similarity()