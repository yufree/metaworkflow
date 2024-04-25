import feedparser
from datetime import datetime, timedelta, timezone
import json
from openai import OpenAI
import requests
import os

# Example PubMed RSS feed URL
rss_url = 'https://pubmed.ncbi.nlm.nih.gov/rss/search/1roWqUHnCMOw0LYZD2R1suracmWnrHoePMKeDCnAP7yHFdPILE/?limit=20&utm_campaign=pubmed-2&fc=20240425120030'

access_token = os.getenv('GITHUB_TOKEN')
openaiapikey = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=openaiapikey
)

def extract_keywords_and_summary(text):
    # Use the OpenAI API to generate keywords and summary
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the text '{text[:4000]}', extract 10 keywords without numbering, separating them by comma and begin the keywords with Keywords:. Additionally, generate a one-sentence summary of the text and begin the keywords with Summary:.Output keywords first then summary",
        max_tokens=150,  # Adjust the length of the generated summary
        n=1,
        stop=None,
        temperature=0.5
    )
    # Parse the API response to extract keywords and summary
    choices = response.choices[0]
    generated_text = choices.text.strip()
    keywords_start_index = generated_text.find("Keywords:")
    summary_start_index = generated_text.find("Summary:")
    keywords = generated_text[keywords_start_index+len("Keywords:"):summary_start_index].strip()
    summary = generated_text[summary_start_index+len("Summary:"):].strip()
    return keywords, summary

def get_pubmed_abstracts(rss_url):
    abstracts_with_urls = []

    # Parse the PubMed RSS feed
    feed = feedparser.parse(rss_url)

    # Calculate the date one week ago
    one_week_ago = datetime.now(timezone.utc) - timedelta(weeks=1)

    # Iterate over entries in the PubMed RSS feed and extract abstracts and URLs
    for entry in feed.entries:
        # Get the publication date of the entry
        published_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')

        # If the publication date is within one week, extract the abstract and URL
        if published_date >= one_week_ago:
            # Get the abstract and URL of the entry
            abstract = entry.content[0].value
            doi = entry.dc_identifier
            abstracts_with_urls.append({"abstract": abstract, "doi": doi})

    return abstracts_with_urls

# Get the abstracts from the PubMed RSS feed
pubmed_abstracts = get_pubmed_abstracts(rss_url)

# Create an empty list to store each abstract with its keywords
new_articles_data = []

for abstract_data in pubmed_abstracts:
    # Extract abstract, keywords, and DOI
    abstract = abstract_data["abstract"]
    keywords, summary = extract_keywords_and_summary(abstract)
    doi = abstract_data["doi"]

    # Build a dictionary containing the abstract, keywords, summary, and DOI
    abstract_data_with_keywords = {
        "abstract": abstract,
        "keywords": keywords,
        "summary": summary,
        "doi": doi
    }

    # Append the dictionary to the list
    new_articles_data.append(abstract_data_with_keywords)

def find_most_similar_sections(new_article_keywords, sections_data, n):
    similar_section_titles = {}
    for article_keywords in new_article_keywords:
        highest_similarity = 0
        most_similar_section_title = None
        new_article_keywords_list = [keyword.strip().lower() for keyword in article_keywords.split(",")]
        new_article_keywords_list = [keyword for keyword in new_article_keywords_list if len(keyword) > 1]
        if new_article_keywords_list:
            for section_title, section_data in sections_data.items():
                section_keywords_list = [keyword.strip().lower() for keyword in section_data["keywords"][0].split(",")]
                section_keywords_list = [keyword for keyword in section_keywords_list if len(keyword) > 1]
                overlap_count = 0
                for keyword in new_article_keywords_list:
                    if keyword in section_keywords_list:
                        overlap_count += 1
                similarity = overlap_count / len(new_article_keywords_list)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_section_title = section_title
        if most_similar_section_title is not None:
            similar_section_titles[article_keywords] = most_similar_section_title
    return similar_section_titles
# Read the merged section data from JSON file
with open('bookkeywords.json', 'r') as file:
    sections_data = json.load(file)

# Create issue title and content
issue_title = f"Weekly Article Matching - {datetime.now().strftime('%Y-%m-%d')}"
issue_body = "Below are the article matching results from the past week:\n\n"

for article_data in new_articles_data:
    abstract = article_data["abstract"]
    keywords = article_data["keywords"].split(", ")
    summary = article_data["summary"]
    doi = article_data.get("doi", "No DOI available")  # Default to "No DOI available" if DOI field is missing

    # Find the most similar section title for each article
    most_similar_section_title = find_most_similar_sections(keywords, sections_data, 1)

    # Check if the most similar section title exists
    if most_similar_section_title:
        # Add the article information to the issue body
        issue_body += f"- Article Abstract: {abstract}\n"
        issue_body += f"  Keywords: {', '.join(keywords)}\n"
        issue_body += f"  Section Title: {most_similar_section_title}\n"
        issue_body += f"  One-sentence Summary: {summary}\n"
        issue_body += f"  DOI: {doi}\n\n"

def create_github_issue(title, body, access_token):
    url = f"https://api.github.com/repos/yufree/metaworkflow/issues"
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "title": title,
        "body": body
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 201:
        print("Issue created successfully!")
    else:
        print("Failed to create issue. Status code:", response.status_code)
        print("Response:", response.text)

# Create the issue
create_github_issue(issue_title, issue_body, access_token)
