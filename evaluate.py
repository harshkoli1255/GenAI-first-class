from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from rag_pipeline import RAGPipeline, get_project_settings


TEST_CASES = [
    {
        "test_id": "precision_verification",
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected_result": "4 attendance marks.",
    },
    {
        "test_id": "multi_hop_reasoning",
        "question": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
        "expected_result": "Email Yashaswini Ma'am within exactly 7 days of the illness or treatment.",
    },
    {
        "test_id": "process_verification",
        "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
        "expected_result": "40% batch support is required, and the proposal must be submitted to the Faculty Coordinator first, not Management.",
    },
    {
        "test_id": "negative_constraint_testing",
        "question": "How much is the fine for smoking a cigarette on campus?",
        "expected_result": "Tobacco is prohibited and leads to Disciplinary Committee action, without inventing a monetary fine.",
    },
]

CSV_PATH = Path(__file__).resolve().parent / "rag_eval_scores.csv"


def _parse_score(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return {
                "score": int(payload.get("score", 0)),
                "reasoning": str(payload.get("reasoning", "No reasoning returned.")),
                "passed": bool(payload.get("passed", False)),
            }
    except json.JSONDecodeError:
        pass

    score_match = re.search(r"score\s*[:=]\s*(\d)", cleaned, flags=re.IGNORECASE)
    passed_match = re.search(r"passed\s*[:=]\s*(true|false|yes|no)", cleaned, flags=re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 0
    passed = str(passed_match.group(1)).lower() in {"true", "yes"} if passed_match else False
    reasoning = cleaned if not score_match else cleaned
    return {"score": score, "reasoning": reasoning, "passed": passed}


def _judge_answer(question: str, expected_result: str, retrieved_context: str, generated_answer: str) -> Dict[str, Any]:
    settings = get_project_settings()
    api_key = settings.get("google_api_key")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing, so the evaluation judge cannot run.")

    judge = ChatGoogleGenerativeAI(
        model=settings.get("google_model", "gemini-3.8-flash"),
        temperature=0.0,
        google_api_key=api_key,
    )
    prompt = f"""
    You are a strict LLM judge for a university policy RAG system.

    Evaluate the generated answer against the official handbook information.
    Rules:
    - Score from 1 to 5.
    - 5 = correct, fully supported by retrieved context, and complies with the no-hallucination rule.
    - 1 = wrong or unsupported.
    - The answer must not claim a policy is present without relevant retrieved context.
    - For negative constraint questions, there should be no invented fine.

    Question: {question}
    Expected result: {expected_result}
    Retrieved context: {retrieved_context}
    Generated answer: {generated_answer}

    Return strict JSON in the form {"score": 1, "reasoning": "short explanation", "passed": true}.
    """
    response = judge.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_score(text)


def _to_row(test: Dict[str, Any], retrieved_context: str, generated_answer: str, score: int, reasoning: str, passed: bool) -> Dict[str, Any]:
    return {
        "test_id": test["test_id"],
        "question": test["question"],
        "expected_result": test["expected_result"],
        "retrieved_context": retrieved_context,
        "generated_answer": generated_answer,
        "score": score,
        "reasoning": reasoning,
        "passed": passed,
    }


def run_evaluation() -> List[Dict[str, Any]]:
    settings = get_project_settings()
    handbook_path = Path(settings.get("handbook_path", "./data/Mirai_SoT_Policy_Handbook_2026.pdf")).resolve()
    if not settings.get("google_api_key") or not handbook_path.exists():
        rows = []
        for test in TEST_CASES:
            rows.append(
                _to_row(
                    test,
                    "not_executed",
                    "not_executed",
                    0,
                    "Evaluation could not run because the Google API key or handbook PDF is missing.",
                    False,
                )
            )
        write_csv(rows)
        print("Evaluation could not run because the handbook PDF or Google API key is missing. CSV was created with not_executed rows.")
        return rows

    pipeline = RAGPipeline()
    try:
        pipeline.ingest_pdf(handbook_path)
    except (FileNotFoundError, ValueError) as exc:
        rows = []
        for test in TEST_CASES:
            rows.append(
                _to_row(
                    test,
                    "not_executed",
                    str(exc),
                    0,
                    "Ingestion failed before evaluation could begin.",
                    False,
                )
            )
        write_csv(rows)
        print(f"Ingestion failed: {exc}")
        return rows

    rows: List[Dict[str, Any]] = []
    for test in TEST_CASES:
        question = test["question"]
        result = pipeline.answer_question(question)
        context = "\n\n---\n\n".join(result.get("retrieved_context", []))
        answer = result.get("answer", "")

        try:
            judge_result = _judge_answer(test["question"], test["expected_result"], context, answer)
            score = int(judge_result.get("score", 0))
            reasoning = str(judge_result.get("reasoning", "No reasoning returned."))
            passed = bool(judge_result.get("passed", False))
        except Exception as exc:
            score = 0
            reasoning = f"Judge failed because: {exc}"
            passed = False

        rows.append(_to_row(test, context, answer, score, reasoning, passed))

    write_csv(rows)
    return rows


def write_csv(rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "test_id",
        "question",
        "expected_result",
        "retrieved_context",
        "generated_answer",
        "score",
        "reasoning",
        "passed",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_evaluation()
