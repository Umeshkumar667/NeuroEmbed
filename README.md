# 🧠 NeuroEmbed

**NeuroEmbed** is a **model-agnostic semantic embedding enrichment framework**.

It does **not replace embedding models**.  
Instead, it **modulates embeddings using semantic context**, producing controlled directional shifts in vector space while preserving dimensionality and normalization.

Designed for:
- RAG systems
- Conversational memory
- Knowledge-aware retrieval
- Agent architectures
- Local / offline-first AI systems

---

## 🚀 Installation

### Standard install (recommended)

```bash
pip install neuroembed


Text Input
   │
   ▼
[ Base Encoder ]
   │
   ▼
Base Embedding  ──────────────┐
                              │
Context Texts ─▶ Encoder ─▶ Context Mean
                              │
                              ▼
                  Context Injector (α)
                              │
                              ▼
                   Enriched Embedding
