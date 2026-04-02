"""
generate_embeddings.py

Generates a semantic embedding index for all book sections.
Run this after each book build to keep the index up-to-date.

Usage:
    python generate_embeddings.py

Requires:
    OPENAI_API_KEY environment variable
    pip install openai
"""

import os
import re
import json
import glob
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
RMD_PATTERN = "[0-9]*.Rmd"
OUTPUT_FILE = "book_embeddings.json"
# Max chars of section content to send for embedding (keeps cost low)
MAX_CONTENT_CHARS = 2000


def parse_rmd_sections(filepath):
    """
    Parse an Rmd file into sections keyed by heading.
    Returns a dict: { "## Section Title": { "content": "...", "chapter_file": "..." } }
    Only extracts ## (h2) and ### (h3) level headings.
    """
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = {}
    current_heading = None
    current_lines = []
    chapter_title = None

    for line in lines:
        # Skip YAML frontmatter
        if line.strip() == "---":
            continue

        # Match # (chapter), ## (section), ### (subsection) headings
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_heading and current_lines:
                content = clean_content("".join(current_lines))
                if len(content.strip()) > 20:  # Skip near-empty sections
                    sections[current_heading] = {
                        "content": content[:MAX_CONTENT_CHARS],
                        "chapter_file": filename,
                        "chapter_title": chapter_title or filename,
                    }

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            if level == 1:
                chapter_title = title
                # Don't create a separate entry for chapter-level heading;
                # its content will flow into the first ## section
                current_heading = None
                current_lines = []
            else:
                current_heading = f"{'#' * level} {title}"
                current_lines = []
            continue

        if current_heading:
            current_lines.append(line)

    # Save last section
    if current_heading and current_lines:
        content = clean_content("".join(current_lines))
        if len(content.strip()) > 20:
            sections[current_heading] = {
                "content": content[:MAX_CONTENT_CHARS],
                "chapter_file": filename,
                "chapter_title": chapter_title or filename,
            }

    return sections


def clean_content(text):
    """Remove R code chunks, HTML tags, and excessive whitespace."""
    # Remove code chunks: ```{r ...} ... ```
    text = re.sub(r"```\{r[^}]*\}.*?```", "", text, flags=re.DOTALL)
    # Remove inline R code
    text = re.sub(r"`r [^`]+`", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove image includes
    text = re.sub(r"knitr::include_graphics\([^)]+\)", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_embedding(text):
    """Get embedding vector from OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000],  # API limit safety
    )
    return response.data[0].embedding


def generate_section_summary(heading, content, chapter_title):
    """
    Build the text to embed: heading + chapter context + content snippet.
    This gives the embedding both structural and semantic information.
    """
    return f"Book chapter: {chapter_title}\nSection: {heading}\n\n{content}"


def main():
    rmd_files = sorted(glob.glob(RMD_PATTERN))
    if not rmd_files:
        print("No Rmd files found. Run this script from the book root directory.")
        return

    print(f"Found {len(rmd_files)} Rmd files")

    # Parse all sections
    all_sections = {}
    for filepath in rmd_files:
        sections = parse_rmd_sections(filepath)
        all_sections.update(sections)
        print(f"  {os.path.basename(filepath)}: {len(sections)} sections")

    print(f"\nTotal sections: {len(all_sections)}")

    # Generate embeddings
    embeddings_data = {}
    texts_to_embed = []
    section_keys = []

    for heading, data in all_sections.items():
        text = generate_section_summary(
            heading, data["content"], data["chapter_title"]
        )
        texts_to_embed.append(text)
        section_keys.append(heading)

    # Batch embedding (API supports batching)
    print(f"Generating embeddings for {len(texts_to_embed)} sections...")
    BATCH_SIZE = 50
    all_embeddings = []
    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} sections embedded")

    # Build output
    for key, embedding in zip(section_keys, all_embeddings):
        data = all_sections[key]
        # Store a short summary (not the full content) to keep file small
        # and to use in the LLM prompt during matching
        short_summary = data["content"][:300].replace("\n", " ").strip()
        embeddings_data[key] = {
            "chapter_file": data["chapter_file"],
            "chapter_title": data["chapter_title"],
            "summary": short_summary,
            "embedding": embedding,
        }

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

    file_size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\nSaved {OUTPUT_FILE} ({file_size_kb:.0f} KB, {len(embeddings_data)} sections)")


if __name__ == "__main__":
    main()
