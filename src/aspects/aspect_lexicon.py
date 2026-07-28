"""Phase 3: seed keyword lexicon for keyword-based aspect tagging.

ASPECT_KEYWORDS maps a business aspect -> seed keywords/lemmas that are matched
against the "clean_deep" column (lowercased, lemmatized, stopwords removed).

This lexicon is a starting point, not a finished product: it was seeded for a
generic e-commerce review domain. When you point this pipeline at a real
dataset, read through the unmatched reviews (i.e. rows with an empty "aspects"
list) and extend the keyword lists below with domain-specific vocabulary
(product names, slang, industry terms, etc.). Keywords should be lemma forms
since they're matched against clean_deep (e.g. "arrive" not "arrived").
"""

ASPECT_KEYWORDS = {
    "delivery": [
        "delivery", "deliver", "ship", "shipping", "arrive", "late",
        "track", "tracking", "dispatch", "courier",
    ],
    "price": [
        "price", "cost", "expensive", "cheap", "value", "overprice",
        "overpriced", "worth", "money",
    ],
    "quality": [
        "quality", "material", "build", "durable", "defective",
        "damage", "damaged", "broken", "break",
    ],
    "support": [
        "support", "service", "response", "respond", "staff", "helpful",
        "rude", "refund", "complaint", "complain",
    ],
    "packaging": [
        "packaging", "package", "box", "wrap", "wrapped", "seal",
        "sealed", "crush", "crushed",
    ],
    "ux": [
        "app", "website", "ui", "interface", "checkout", "navigation",
        "navigate", "easy", "confusing", "confuse",
    ],
}
