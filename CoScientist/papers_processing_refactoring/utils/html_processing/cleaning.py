from bs4 import BeautifulSoup, Tag

BLACKLIST_HEADERS = [
    "author information", "associated content", "acknowledgment", "acknowledgement",
    "acknowledgments", "acknowledgements", "references", "data availability",
    "declaration of competing interest", "credit authorship contribution statement",
    "funding", "ethical statements", "supplementary materials",
    "conflict of interest", "conflicts of interest",
    "author contributions", "data availability statement",
    "ethics approval", "supplementary information"
]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for header in soup.find_all(["h1", "h2", "h3"]):
        header_text = header.get_text(strip=True).lower()

        if any(exclude in header_text for exclude in BLACKLIST_HEADERS):
            next_node = header.next_sibling
            elements_to_remove = []

            while next_node and getattr(next_node, "name", None) not in ["h1", "h2", "h3"]:
                elements_to_remove.append(next_node)
                next_node = next_node.next_sibling

            header.decompose()
            for el in elements_to_remove:
                if isinstance(el, Tag):
                    el.decompose()

    return soup.prettify()
