from neuroembed.core import NeuroEmbed
from neuroembed.encoders.sentence_transformer import SentenceTransformerEncoder
from numpy.linalg import norm

encoder = SentenceTransformerEncoder()
ne = NeuroEmbed(encoder=encoder, alpha=0.6)

query = "bank interest rate"

context = [
    "RBI monetary policy",
    "repo rate",
    "inflation control"
]

plain = encoder.encode([query])[0]
enriched = ne.embed(query, context)

print("Plain embedding norm:", norm(plain))
print("Enriched embedding norm:", norm(enriched))
print("Cosine similarity (plain vs enriched):", plain @ enriched)
