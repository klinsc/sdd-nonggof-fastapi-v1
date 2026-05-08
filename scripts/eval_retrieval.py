"""Threshold-tuning eval harness for the RAG retrieval pipeline.

Sends a labeled set of queries through `ChromaVectorStoreRepository`
(retrieval only — no LLM), then sweeps `RETRIEVAL_SCORE_THRESHOLD`
values and reports precision / recall / F1 per threshold.

Run with:
    $env:PYTHONPATH = "."; $env:OLLAMA_HOST = "http://127.0.0.1:11434"
    python scripts/eval_retrieval.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import get_settings
from app.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddingsProvider
from app.infrastructure.vectorstore.chroma_repo import ChromaVectorStoreRepository

Label = Literal["RETRIEVE", "EDGE", "REFUSE"]


@dataclass
class Query:
    text: str
    label: Label
    note: str = ""


# Queries derived from inspecting data/sdd-data and Chroma sample chunks.
# RETRIEVE: phrasings that directly match indexed content
# EDGE: domain-related but topic may not be covered by these approval docs
# REFUSE: off-topic — the threshold should reject these
QUERIES: list[Query] = [
    # --- SHOULD_RETRIEVE ---
    Query("หม้อแปลงไฟฟ้ากำลัง 115-22 kV ขนาด 50 MVA", "RETRIEVE", "transformer specs"),
    Query("การก่อสร้างสถานีไฟฟ้าคลองหนึ่ง", "RETRIEVE", "specific substation"),
    Query("เพิ่ม Bay line ที่สถานีไฟฟ้าวังม่วง", "RETRIEVE", "specific Bay line file"),
    Query("switchgear ชนิด MTS Outdoor Type ระบบ 115 kV", "RETRIEVE", "switchgear spec"),
    Query("การจัดบัสแบบ H-Configuration", "RETRIEVE", "bus configuration"),
    Query("ค่าใช้จ่ายในการก่อสร้างสถานีไฟฟ้า", "RETRIEVE", "cost breakdown"),
    Query("Riser pole ระบบ 22 เควี", "RETRIEVE", "riser pole 22kV"),
    Query("อาคารควบคุม 2 ชั้น สถานีไฟฟ้า", "RETRIEVE", "control building"),

    # --- EDGE (domain-related, may or may not be indexed) ---
    Query("การคำนวณกระแสลัดวงจร 115 kV", "EDGE", "short-circuit calc"),
    Query("มาตรฐานการป้องกันฟ้าผ่าในสถานีไฟฟ้า", "EDGE", "lightning protection"),
    Query("ระยะห่างทางไฟฟ้าใน switchyard", "EDGE", "clearance distances"),
    Query("การออกแบบระบบ grounding ของสถานีไฟฟ้า", "EDGE", "grounding"),
    Query("การติดตั้ง CT และ PT", "EDGE", "instrument transformers"),

    # --- SHOULD_REFUSE (off-topic) ---
    Query("วิธีปลูกข้าวให้ได้ผลผลิตสูง", "REFUSE", "rice farming"),
    Query("ราคาทองคำวันนี้", "REFUSE", "gold price"),
    Query("สูตรทำต้มยำกุ้ง", "REFUSE", "tom yum recipe"),
    Query("การพัฒนาแอปพลิเคชันมือถือด้วย React Native", "REFUSE", "mobile dev"),
    Query("How to fix a leaky faucet", "REFUSE", "English plumbing"),
    Query("ความหมายของคำว่า ปรัชญา", "REFUSE", "philosophy"),
    Query("วิธีเลี้ยงสุนัข", "REFUSE", "dog raising"),
]

K = 6  # matches production RETRIEVAL_K
THRESHOLDS = [0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]


def main() -> None:
    s = get_settings()
    emb = OllamaEmbeddingsProvider(s).build()
    repo = ChromaVectorStoreRepository(s, emb)
    handle = repo.open()

    # Run all queries once; keep top scores for analysis.
    rows = []
    for q in QUERIES:
        results = handle.similarity_search_with_score(q.text, k=K)
        scores = [score for _, score in results]
        top_score = scores[0] if scores else float("inf")
        top_src = results[0][0].metadata.get("source", "?") if results else ""
        rows.append({
            "query": q,
            "scores": scores,
            "top_score": top_score,
            "top_src": top_src,
        })

    # --- Per-query dump ---
    print("\n=== PER-QUERY RESULTS (k=%d) ===\n" % K)
    for r in rows:
        q = r["query"]
        scores_str = ", ".join(f"{s:.3f}" for s in r["scores"])
        print(f"[{q.label:<8}] top={r['top_score']:.3f}  «{q.text}»")
        print(f"           ({q.note})")
        print(f"           all_scores=[{scores_str}]")
        print(f"           top_src={r['top_src']}")
        print()

    # --- Threshold sweep ---
    print("=== THRESHOLD SWEEP ===")
    print("(Counted: RETRIEVE & REFUSE only. EDGE shown separately.)\n")
    header = f"{'thr':>5}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  {'precision':>9}  {'recall':>6}  {'F1':>5}"
    print(header)
    print("-" * len(header))

    best = (None, -1.0)  # (threshold, f1)
    for thr in THRESHOLDS:
        tp = fp = fn = tn = 0
        for r in rows:
            q = r["query"]
            passes = r["top_score"] <= thr  # at least the best chunk passes
            if q.label == "RETRIEVE":
                tp += 1 if passes else 0
                fn += 0 if passes else 1
            elif q.label == "REFUSE":
                fp += 1 if passes else 0
                tn += 0 if passes else 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        print(f"{thr:>5.2f}  {tp:>3} {fp:>3} {fn:>3} {tn:>3}  {prec:>9.3f}  {recall:>6.3f}  {f1:>5.3f}")
        if f1 > best[1]:
            best = (thr, f1)

    # --- Edge query report ---
    print("\n=== EDGE QUERIES (info only — verify by hand) ===")
    for r in rows:
        q = r["query"]
        if q.label == "EDGE":
            print(f"  top={r['top_score']:.3f}  «{q.text}»  ({q.note})")
            print(f"           top_src={r['top_src']}")

    # --- Recommendation ---
    print("\n=== RECOMMENDATION ===")
    if best[0] is not None:
        print(f"Best F1 = {best[1]:.3f} at threshold {best[0]:.2f}")
    print(f"Current setting: {s.RETRIEVAL_SCORE_THRESHOLD}")


if __name__ == "__main__":
    main()
