from typing import Literal, TypeAlias


ResearchDomain: TypeAlias = Literal[
    "Physical Sciences",
    "Social Sciences",
    "Life Sciences",
    "Health Sciences",
]


DOMAIN_TO_SUBDOMAINS: dict[str, list[str]] = {
    "Physical Sciences": [
        "Earth and Planetary Sciences",
        "Physics and Astronomy",
        "Environmental Science",
        "Chemistry",
        "Engineering",
        "Computer Science",
        "Energy",
        "Mathematics",
        "Materials Science",
        "Chemical Engineering"
    ],
    "Social Sciences": [
        "Business, Management and Accounting",
        "Economics, Econometrics and Finance",
        "Social Sciences",
        "Arts and Humanities",
        "Decision Sciences",
        "Psychology"
    ],
    "Life Sciences": [
        "Agricultural and Biological Sciences",
        "Biochemistry, Genetics and Molecular Biology",
        "Immunology and Microbiology",
        "Neuroscience",
        "Pharmacology, Toxicology and Pharmaceutics"
    ],
    "Health Sciences": [
        "Medicine",
        "Dentistry",
        "Health Professions",
        "Nursing",
        "Veterinary"
    ],
}


def get_research_domains() -> list[str]:
    return list(DOMAIN_TO_SUBDOMAINS.keys())


def get_sub_domains_for_domain(domain: str | None) -> list[str]:
    if not domain:
        return []
    return DOMAIN_TO_SUBDOMAINS.get(domain, [])


def format_domain_subdomain_mapping_for_prompt() -> str:
    lines: list[str] = []
    for domain, subdomains in DOMAIN_TO_SUBDOMAINS.items():
        if subdomains:
            lines.append(f"- {domain}: {', '.join(subdomains)}")
        else:
            lines.append(f"- {domain}: no predefined sub-domains")
    return "\\n".join(lines)
