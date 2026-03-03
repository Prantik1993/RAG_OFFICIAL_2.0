"""
RAGAS Evaluation Suite  (v4.2)
==============================
Runs the 20-question GDPR test set through the live RAG pipeline
and measures 4 RAGAS metrics:

  Faithfulness      — answer only uses retrieved context (no hallucination)
  Answer Relevancy  — answer actually addresses the question
  Context Precision — retrieved docs are relevant to the question
  Context Recall    — retrieved docs cover the ground truth

Usage:
    python -m src.evaluation.ragas_eval
    python -m src.evaluation.ragas_eval --version 2      # test prompt v2
    python -m src.evaluation.ragas_eval --quick          # first 5 questions only
    python -m src.evaluation.ragas_eval --compare 1 2    # compare two prompt versions

Output:
    evaluation/results_vN_TIMESTAMP.json
    evaluation/latest_results.json  (always overwritten)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import src.config as cfg
from src.logger import get_logger

log = get_logger("Evaluation")

_TESTSET_PATH  = Path(__file__).resolve().parent.parent.parent / "evaluation" / "gdpr_testset.json"
_RESULTS_DIR   = cfg.EVAL_DIR
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Metrics (lightweight, no ragas dependency required) ───────────────────────
def _faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Heuristic: what fraction of sentences in the answer contain terms
    from the retrieved context?  True RAGAS uses an LLM judge — this
    approximation runs offline with no extra API cost.
    """
    if not contexts or not answer:
        return 0.0
    combined = " ".join(contexts).lower()
    sentences = [s.strip() for s in answer.replace("?", ".").replace("!", ".").split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    grounded = 0
    for sent in sentences:
        words = [w for w in sent.lower().split() if len(w) > 4]
        hits  = sum(1 for w in words if w in combined)
        if words and hits / len(words) >= 0.3:
            grounded += 1
    return round(grounded / len(sentences), 3)


def _answer_relevancy(answer: str, question: str) -> float:
    """
    Heuristic: key noun/verb overlap between question and answer.
    True RAGAS generates questions from the answer and measures similarity.
    """
    if not answer or not question:
        return 0.0
    stopwords = {"what","is","the","a","an","of","in","for","and","or","are","does","do","how","when","why","which"}
    q_words = {w.lower().strip("?,") for w in question.split() if w.lower() not in stopwords and len(w) > 3}
    a_words = {w.lower().strip("?.,:") for w in answer.split() if len(w) > 3}
    if not q_words:
        return 0.5
    return round(min(1.0, len(q_words & a_words) / len(q_words)), 3)


def _context_precision(contexts: list[str], ground_truth: str) -> float:
    """What fraction of retrieved contexts are relevant to the ground truth?"""
    if not contexts:
        return 0.0
    gt_words = set(ground_truth.lower().split())
    relevant = 0
    for ctx in contexts:
        ctx_words = set(ctx.lower().split())
        overlap   = len(gt_words & ctx_words) / max(len(gt_words), 1)
        if overlap > 0.15:
            relevant += 1
    return round(relevant / len(contexts), 3)


def _context_recall(contexts: list[str], ground_truth: str) -> float:
    """What fraction of ground truth is covered by retrieved contexts?"""
    if not contexts or not ground_truth:
        return 0.0
    combined  = " ".join(contexts).lower()
    gt_words  = [w for w in ground_truth.lower().split() if len(w) > 4]
    if not gt_words:
        return 0.0
    covered   = sum(1 for w in gt_words if w in combined)
    return round(covered / len(gt_words), 3)


# ── Evaluator ─────────────────────────────────────────────────────────────────
class RAGASEvaluator:

    def __init__(self, engine) -> None:
        self._engine = engine

    def run(
        self,
        prompt_version: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        testset = self._load_testset()
        if limit:
            testset = testset[:limit]

        prompt_cfg = self._engine.prompt_registry.get(prompt_version)
        log.info(f"Evaluating prompt v{prompt_cfg.version} on {len(testset)} questions…")

        results     = []
        faiths      = []
        relevancies = []
        precisions  = []
        recalls     = []

        for i, item in enumerate(testset, 1):
            q  = item["question"]
            gt = item["ground_truth"]
            log.info(f"  [{i}/{len(testset)}] {item['id']}: {q[:60]}…")

            t0 = time.perf_counter()
            try:
                resp = self._engine.query(q, prompt_version=prompt_cfg.version)
            except Exception as exc:
                log.error(f"Query failed: {exc}")
                continue

            latency = (time.perf_counter() - t0) * 1000
            answer  = resp.get("answer", "")
            ctx_docs = resp.get("context", [])
            contexts = [d.page_content for d in ctx_docs]

            faith    = _faithfulness(answer, contexts)
            relev    = _answer_relevancy(answer, q)
            prec     = _context_precision(contexts, gt)
            recall   = _context_recall(contexts, gt)

            faiths.append(faith)
            relevancies.append(relev)
            precisions.append(prec)
            recalls.append(recall)

            results.append({
                "id":              item["id"],
                "question":        q,
                "reference":       item.get("reference"),
                "ground_truth":    gt,
                "answer":          answer,
                "latency_ms":      round(latency),
                "faithfulness":    faith,
                "answer_relevancy": relev,
                "context_precision": prec,
                "context_recall":  recall,
                "num_contexts":    len(contexts),
            })

        avg = lambda lst: round(sum(lst) / len(lst), 3) if lst else 0

        summary = {
            "prompt_version":     prompt_cfg.version,
            "prompt_description": prompt_cfg.description,
            "model":              prompt_cfg.model,
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "total_questions":    len(results),
            "metrics": {
                "faithfulness":      avg(faiths),
                "answer_relevancy":  avg(relevancies),
                "context_precision": avg(precisions),
                "context_recall":    avg(recalls),
                "avg_latency_ms":    round(sum(r["latency_ms"] for r in results) / max(len(results), 1)),
            },
            "per_question": results,
        }

        self._save(summary)
        self._print_summary(summary)
        return summary

    def compare(self, version_a: str, version_b: str) -> dict:
        """Run eval on both prompt versions and print a diff table."""
        log.info(f"Comparing prompt v{version_a} vs v{version_b}…")
        res_a = self.run(prompt_version=version_a)
        res_b = self.run(prompt_version=version_b)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "diff": {}
        }
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            a = res_a["metrics"][metric]
            b = res_b["metrics"][metric]
            comparison["diff"][metric] = {
                "v" + version_a: a,
                "v" + version_b: b,
                "delta": round(b - a, 3),
                "winner": f"v{version_b}" if b > a else f"v{version_a}" if a > b else "tie",
            }

        path = _RESULTS_DIR / f"comparison_v{version_a}_vs_v{version_b}.json"
        with open(path, "w") as f:
            json.dump(comparison, f, indent=2)

        log.info(f"Comparison saved: {path}")
        print("\n── Comparison Summary ─────────────────────────────")
        for metric, d in comparison["diff"].items():
            delta_str = f"+{d['delta']}" if d['delta'] > 0 else str(d['delta'])
            print(f"  {metric:<25} v{version_a}={d['v'+version_a]:.3f}  v{version_b}={d['v'+version_b]:.3f}  Δ={delta_str}  winner={d['winner']}")
        return comparison

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load_testset() -> list[dict]:
        with open(_TESTSET_PATH, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, summary: dict) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _RESULTS_DIR / f"results_v{summary['prompt_version']}_{ts}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        latest = _RESULTS_DIR / "latest_results.json"
        with open(latest, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Results saved: {path}")

    @staticmethod
    def _print_summary(summary: dict) -> None:
        m = summary["metrics"]
        print(f"""
╔══════════════════════════════════════════════════╗
║  RAGAS Evaluation — Prompt v{summary['prompt_version']:<21}║
╠══════════════════════════════════════════════════╣
║  Faithfulness       {m['faithfulness']:.3f}                        ║
║  Answer Relevancy   {m['answer_relevancy']:.3f}                        ║
║  Context Precision  {m['context_precision']:.3f}                        ║
║  Context Recall     {m['context_recall']:.3f}                        ║
║  Avg Latency        {m['avg_latency_ms']:>5} ms                      ║
║  Questions          {summary['total_questions']:<5}                         ║
╚══════════════════════════════════════════════════╝
        """)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--version", default=None, help="Prompt version to evaluate")
    parser.add_argument("--quick",   action="store_true", help="First 5 questions only")
    parser.add_argument("--compare", nargs=2, metavar=("V1", "V2"), help="Compare two prompt versions")
    args = parser.parse_args()

    # Bootstrap engine (needs running FAISS index)
    from src.ingestion.pipeline import IngestionPipeline
    from src.rag.engine import RAGEngine
    from src.vector_store.manager import VectorStoreManager

    vs_mgr   = VectorStoreManager()
    pipeline = IngestionPipeline()
    vs, bm25 = vs_mgr.load_or_create(pipeline.run)
    engine   = RAGEngine(vs, bm25)

    evaluator = RAGASEvaluator(engine)

    if args.compare:
        evaluator.compare(args.compare[0], args.compare[1])
    else:
        evaluator.run(
            prompt_version=args.version,
            limit=5 if args.quick else None,
        )


if __name__ == "__main__":
    main()
