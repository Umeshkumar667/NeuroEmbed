import numpy as np
from numpy.linalg import norm
from neuroembed.core import NeuroEmbed
from neuroembed.encoders.sentence_transformer import SentenceTransformerEncoder


def cosine(a, b):
    return float(a @ b)


def run_benchmark():
    encoder = SentenceTransformerEncoder()
    ne = NeuroEmbed(encoder=encoder, alpha=0.6)

    query = "bank interest rate"

    contexts = {
        "finance_context": [
            "RBI monetary policy",
            "repo rate",
            "inflation control",
            "central bank decision"
        ],
        "river_context": [
            "river bank erosion",
            "flood plains",
            "soil conservation",
            "water flow"
        ],
        "no_context": None
    }

    print("\n=== NeuroEmbed Semantic Shift Benchmark ===\n")

    base = encoder.encode([query])[0]
    print(f"Base embedding norm: {norm(base):.4f}\n")

    for name, ctx in contexts.items():
        enriched = ne.embed(query, ctx)

        print(f"Context: {name}")
        print(f"  Cosine(base, enriched): {cosine(base, enriched):.4f}")
        print(f"  Enriched norm: {norm(enriched):.4f}")
        print("-" * 45)


if __name__ == "__main__":
    run_benchmark()
