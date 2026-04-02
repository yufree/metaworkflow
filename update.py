"""
update.py

Weekly article matching pipeline for metaworkflow book.
1. Fetches new articles from PubMed RSS feed
2. Matches them to book sections using embedding similarity + LLM verification
3. Generates draft content and creates a GitHub PR (or falls back to an issue)

Usage:
    python update.py

Requires environment variables:
    OPENAI_API_KEY
    GITHUB_TOKEN (with repo write access)

Requires:
    pip install feedparser requests openai numpy
"""

import feedparser
from datetime import datetime, timedelta, timezone
import json
import requests
import os
import re
import numpy as np
import openai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RSS_URL = (
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/"
    "1roWqUHnCMOw0LYZD2R1suracmWnrHoePMKeDCnAP7yHFdPILE/"
    "?limit=20&utm_campaign=pubmed-2&fc=20240425120030"
)

REPO = "yufree/metaworkflow"
EMBEDDINGS_FILE = "book_embeddings.json"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
TOP_K = 3  # Number of candidate sections for LLM to choose from
SIMILARITY_THRESHOLD = 0.5  # Minimum cosine similarity to consider a match

access_token = os.getenv("GITHUB_TOKEN")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def get_embedding(text):
    """Get embedding vector from OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000],
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def load_book_embeddings():
    """Load pre-computed section embeddings."""
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert embedding lists to numpy arrays
    for key in data:
        data[key]["embedding"] = np.array(data[key]["embedding"])
    return data


def find_top_sections(article_embedding, book_embeddings, top_k=TOP_K):
    """
    Find the top-k most similar book sections by cosine similarity.
    Returns list of (section_heading, similarity_score, section_data).
    """
    scores = []
    for heading, section_data in book_embeddings.items():
        sim = cosine_similarity(article_embedding, section_data["embedding"])
        scores.append((heading, sim, section_data))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# LLM calls (minimal token usage)
# ---------------------------------------------------------------------------


def llm_match_and_summarize(abstract, doi, top_sections):
    """
    Two-in-one LLM call: pick the best section AND generate a one-sentence
    summary + draft insert text. This saves one API call vs doing them separately.

    Input tokens: ~800 (3 section titles/summaries + abstract)
    Output tokens: ~300
    Cost per article: < $0.001 with gpt-4o-mini
    """
    section_options = "\n".join(
        f"  {i+1}. {heading} (Chapter: {data['chapter_title']})\n"
        f"     Context: {data['summary'][:150]}"
        for i, (heading, score, data) in enumerate(top_sections)
    )

    prompt = f"""You are a metabolomics expert maintaining an online textbook.

A new article was published:
DOI: {doi}
Abstract: {abstract[:2000]}

The top candidate book sections for this article are:
{section_options}
  {len(top_sections)+1}. NONE - this article does not fit any of the above sections

Tasks:
1. Pick the BEST matching section number (or NONE if similarity < 0.5).
2. Write a one-sentence summary of the article's contribution.
3. Write a 1-2 sentence draft text that could be inserted into the matched section,
   in the same style as the book (informative, concise, with a citation placeholder).
   Use this format for the citation: [@NEWREF_{doi.split('/')[-1][:20] if doi else 'unknown'}]

Respond in this exact JSON format:
{{"match": <number or "NONE">, "summary": "...", "draft": "..."}}"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()
    # Clean potential markdown code fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"match": "NONE", "summary": text, "draft": ""}

    return result


# ---------------------------------------------------------------------------
# RSS feed parsing
# ---------------------------------------------------------------------------


def get_pubmed_abstracts(rss_url):
    """Fetch abstracts from the past week via PubMed RSS."""
    feed = feedparser.parse(rss_url)
    one_week_ago = datetime.now(timezone.utc) - timedelta(weeks=1)
    articles = []

    for entry in feed.entries:
        try:
            published_date = datetime.strptime(
                entry.published, "%a, %d %b %Y %H:%M:%S %z"
            )
        except (ValueError, AttributeError):
            continue

        if published_date >= one_week_ago:
            abstract = entry.content[0].value if entry.get("content") else ""
            doi = getattr(entry, "dc_identifier", "")
            title = getattr(entry, "title", "")
            articles.append(
                {"title": title, "abstract": abstract, "doi": doi}
            )

    return articles


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def github_api(method, endpoint, data=None):
    """Generic GitHub API helper."""
    url = f"https://api.github.com/repos/{REPO}{endpoint}"
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    if method == "GET":
        resp = requests.get(url, headers=headers)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data)
    elif method == "PUT":
        resp = requests.put(url, headers=headers, json=data)
    elif method == "PATCH":
        resp = requests.patch(url, headers=headers, json=data)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return resp


def create_github_issue(title, body):
    """Create a GitHub issue as fallback."""
    resp = github_api("POST", "/issues", {"title": title, "body": body})
    if resp.status_code == 201:
        print(f"Issue created: {resp.json()['html_url']}")
    else:
        print(f"Failed to create issue: {resp.status_code} {resp.text}")


def create_github_pr(title, body, branch_name, file_changes):
    """
    Create a PR with file changes.
    file_changes: list of {"path": str, "content": str}
    """
    # 1. Get the SHA of the default branch (master)
    resp = github_api("GET", "/git/ref/heads/master")
    if resp.status_code != 200:
        print(f"Failed to get master ref: {resp.text}")
        return None
    base_sha = resp.json()["object"]["sha"]

    # 2. Create a new branch
    resp = github_api(
        "POST",
        "/git/refs",
        {"ref": f"refs/heads/{branch_name}", "sha": base_sha},
    )
    if resp.status_code != 201:
        # Branch might already exist, try to update it
        resp = github_api(
            "PATCH",
            f"/git/refs/heads/{branch_name}",
            {"sha": base_sha, "force": True},
        )
        if resp.status_code != 200:
            print(f"Failed to create/update branch: {resp.text}")
            return None

    # 3. For each file, get current content and update
    for change in file_changes:
        # Get current file to obtain its SHA
        resp = github_api(
            "GET", f"/contents/{change['path']}?ref={branch_name}"
        )
        file_sha = resp.json().get("sha", "") if resp.status_code == 200 else ""

        import base64

        encoded = base64.b64encode(change["content"].encode("utf-8")).decode()
        update_data = {
            "message": f"Add weekly article update to {change['path']}",
            "content": encoded,
            "branch": branch_name,
        }
        if file_sha:
            update_data["sha"] = file_sha

        resp = github_api("PUT", f"/contents/{change['path']}", update_data)
        if resp.status_code not in (200, 201):
            print(f"Failed to update {change['path']}: {resp.text}")
            return None

    # 4. Create PR
    resp = github_api(
        "POST",
        "/pulls",
        {
            "title": title,
            "body": body,
            "head": branch_name,
            "base": "master",
        },
    )
    if resp.status_code == 201:
        pr_url = resp.json()["html_url"]
        print(f"PR created: {pr_url}")
        return pr_url
    else:
        print(f"Failed to create PR: {resp.status_code} {resp.text}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    # Load book embeddings
    print("Loading book embeddings...")
    try:
        book_embeddings = load_book_embeddings()
    except FileNotFoundError:
        print(
            f"ERROR: {EMBEDDINGS_FILE} not found. "
            "Run generate_embeddings.py first."
        )
        return
    print(f"Loaded {len(book_embeddings)} sections")

    # Fetch new articles
    print("Fetching PubMed RSS feed...")
    articles = get_pubmed_abstracts(RSS_URL)
    print(f"Found {len(articles)} articles from past week")

    if not articles:
        print("No new articles found. Exiting.")
        return

    # Process each article
    matched_articles = []  # Articles matched to sections
    unmatched_articles = []  # Articles that don't fit any section

    for i, article in enumerate(articles):
        print(f"\nProcessing article {i+1}/{len(articles)}: {article['doi']}")

        # Step 1: Embed the abstract
        article_emb = get_embedding(article["abstract"][:4000])

        # Step 2: Find top-k candidate sections by cosine similarity
        top_sections = find_top_sections(article_emb, book_embeddings)
        best_sim = top_sections[0][1] if top_sections else 0

        print(
            f"  Top match: {top_sections[0][0]} (sim={top_sections[0][1]:.3f})"
        )

        # Step 3: Skip LLM call if similarity is very low
        if best_sim < SIMILARITY_THRESHOLD:
            print("  Skipping LLM call (similarity too low)")
            unmatched_articles.append(
                {
                    "article": article,
                    "best_section": top_sections[0][0],
                    "similarity": best_sim,
                }
            )
            continue

        # Step 4: LLM match + draft generation (single API call)
        result = llm_match_and_summarize(
            article["abstract"], article["doi"], top_sections
        )

        match_idx = result.get("match")
        if match_idx == "NONE" or match_idx is None:
            unmatched_articles.append(
                {
                    "article": article,
                    "best_section": top_sections[0][0],
                    "similarity": best_sim,
                    "summary": result.get("summary", ""),
                }
            )
            print("  LLM says: no good match")
        else:
            try:
                idx = int(match_idx) - 1
                heading, sim, section_data = top_sections[idx]
            except (ValueError, IndexError):
                heading, sim, section_data = top_sections[0]

            matched_articles.append(
                {
                    "article": article,
                    "section_heading": heading,
                    "chapter_file": section_data["chapter_file"],
                    "chapter_title": section_data["chapter_title"],
                    "similarity": sim,
                    "summary": result.get("summary", ""),
                    "draft": result.get("draft", ""),
                }
            )
            print(f"  Matched to: {heading} | Draft ready")

    # Build output
    date_str = datetime.now().strftime("%Y-%m-%d")

    if matched_articles:
        # Try to create a PR with the draft changes
        print(f"\n--- Creating PR with {len(matched_articles)} matched articles ---")

        # Group matched articles by chapter file
        changes_by_file = {}
        for match in matched_articles:
            filepath = match["chapter_file"]
            if filepath not in changes_by_file:
                changes_by_file[filepath] = []
            changes_by_file[filepath].append(match)

        # Read current file contents and append drafts
        file_changes = []
        pr_body_parts = ["## Weekly Article Update\n"]

        for filepath, matches in changes_by_file.items():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"  WARNING: {filepath} not found, skipping")
                continue

            additions = []
            for match in matches:
                draft = match["draft"]
                doi = match["article"]["doi"]
                heading = match["section_heading"]
                summary = match["summary"]

                # Find the section and append after it
                additions.append(
                    {
                        "heading": heading,
                        "text": draft,
                        "doi": doi,
                        "summary": summary,
                    }
                )

                pr_body_parts.append(
                    f"- **{heading}** ({filepath}): {summary}\n"
                    f"  - DOI: {doi}\n"
                )

            # Insert draft text after each matched section
            new_content = content
            for addition in additions:
                heading = addition["heading"]
                text_to_add = f"\n{addition['text']}\n"

                # Find the heading in the file
                pattern = re.escape(heading)
                heading_match = re.search(
                    f"^{pattern}$", new_content, re.MULTILINE
                )
                if heading_match:
                    # Find the next heading of same or higher level
                    level = heading.count("#")
                    next_heading = re.search(
                        f"^#{{1,{level}}} ",
                        new_content[heading_match.end() :],
                        re.MULTILINE,
                    )
                    if next_heading:
                        insert_pos = heading_match.end() + next_heading.start()
                    else:
                        insert_pos = len(new_content)

                    # Insert before the next heading (with a blank line)
                    new_content = (
                        new_content[:insert_pos].rstrip()
                        + "\n\n"
                        + text_to_add.strip()
                        + "\n\n"
                        + new_content[insert_pos:]
                    )

            if new_content != content:
                file_changes.append({"path": filepath, "content": new_content})

        if file_changes:
            branch_name = f"weekly-update-{date_str}"
            pr_body = "\n".join(pr_body_parts)
            pr_body += (
                "\n\n---\n"
                "*Auto-generated by weekly article matching pipeline. "
                "Please review the draft text before merging.*\n"
            )

            pr_url = create_github_pr(
                title=f"Weekly Article Update - {date_str}",
                body=pr_body,
                branch_name=branch_name,
                file_changes=file_changes,
            )

            if not pr_url:
                # Fallback to issue if PR creation fails
                print("PR creation failed, falling back to issue...")
                _create_fallback_issue(
                    date_str, matched_articles, unmatched_articles
                )
        else:
            print("No file changes to make")
    else:
        print("No articles matched to book sections")

    # Always create an issue if there are unmatched articles (potential new topics)
    if unmatched_articles:
        _create_new_topics_issue(date_str, unmatched_articles)


def _create_fallback_issue(date_str, matched_articles, unmatched_articles):
    """Create an issue with all results when PR creation fails."""
    body = "## Matched Articles\n\n"
    for match in matched_articles:
        body += f"### {match['section_heading']}\n"
        body += f"- **DOI**: {match['article']['doi']}\n"
        body += f"- **Summary**: {match['summary']}\n"
        body += f"- **Similarity**: {match['similarity']:.3f}\n"
        body += f"- **Draft text**:\n\n> {match['draft']}\n\n"

    if unmatched_articles:
        body += "## Unmatched Articles (potential new topics)\n\n"
        for item in unmatched_articles:
            body += f"- DOI: {item['article']['doi']}\n"
            body += f"  Closest section: {item['best_section']} "
            body += f"(sim={item['similarity']:.3f})\n"

    create_github_issue(f"Weekly Article Matching - {date_str}", body)


def _create_new_topics_issue(date_str, unmatched_articles):
    """Create an issue listing articles that don't fit existing sections."""
    body = (
        "The following articles from this week did not match any existing "
        "book section well. They may represent new topics worth adding:\n\n"
    )
    for item in unmatched_articles:
        doi = item["article"]["doi"]
        title = item["article"].get("title", "")
        best = item["best_section"]
        sim = item["similarity"]
        summary = item.get("summary", "")

        body += f"- **{title}**\n"
        body += f"  - DOI: {doi}\n"
        body += f"  - Closest section: {best} (sim={sim:.3f})\n"
        if summary:
            body += f"  - Summary: {summary}\n"
        body += "\n"

    create_github_issue(
        f"New Topics Detected - {date_str}",
        body,
    )


if __name__ == "__main__":
    main()
