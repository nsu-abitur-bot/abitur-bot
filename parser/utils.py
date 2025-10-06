def parse_header_faculty(soup, data):
    title = soup.title.text if soup.title else "Не найдено"
    data["title"] = title

    meta_desc = soup.find("meta", attrs={"name": "description"})
    data["description"] = meta_desc["content"] if meta_desc else ""

    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    data["keywords"] = meta_keywords["content"] if meta_keywords else ""

def parse_content_blocks(soup, data, style, attr):
    content_blocks = soup.select(f"{style}.{attr}")
    data["content_blocks"] = [block.text for block in content_blocks if
                              block.text and block.text.strip()] if content_blocks else []