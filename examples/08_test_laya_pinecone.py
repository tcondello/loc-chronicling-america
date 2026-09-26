#!/usr/bin/env python3
"""Example 08: Test Laya Decision Engine on Real Pinecone Chunks.

Fetches random historical newspaper text chunks from a Pinecone Document Schema index,
runs Laya System 1 non-autoregressive decisions (NER entity presence, topic classification,
editorial intent, and historical scoring) in a single forward pass, and formats the output.

Usage:
    # Test random chunk from a random namespace
    python examples/08_test_laya_pinecone.py

    # Test random chunk from a specific year/namespace
    python examples/08_test_laya_pinecone.py --namespace 1906

    # Test an exact document ID
    python examples/08_test_laya_pinecone.py --namespace 1891 --doc-id beatrice#1891-01-02#ed-1#p03#c16

    # Test custom text without fetching from Pinecone
    python examples/08_test_laya_pinecone.py --text "Governor Mickey arrived in Lincoln today to address the state legislature."

    # Interactive continuous loop (press Enter to test next random chunk)
    python examples/08_test_laya_pinecone.py --loop

    # Select question preset: 'full', 'ner', 'classification', 'triage', or 'ocr'
    python examples/08_test_laya_pinecone.py --preset ocr
"""

import argparse
import itertools
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress known non-critical temperature calibration warning emitted by laya 0.3.x checkpoints
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*uncalibrated.*")

try:
    from dotenv import find_dotenv, load_dotenv
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

try:
    from laya import Router
except ImportError:
    Router = None


# Pre-defined Question Suites for Historical Newspapers & General Documents
QUESTION_PRESETS = {
    "full": {
        "topic": {
            "type": "choice",
            "instructions": "What is the primary topic of this newspaper excerpt?",
            "criteria": {
                "politics_government": "Elections, state affairs, legislation, and political leaders",
                "crime_courts": "Legal trials, crimes, police activity, and judicial verdicts",
                "business_markets": "Commodity prices, trade, railroad reports, and financial news",
                "society_local": "Local community updates, social notices, obituaries, and events",
                "advertisement": "Classified advertisements, product pitches, remedies, and notices",
                "general_news": "General news or miscellaneous reporting"
            }
        },
        "primary_entity": {
            "type": "choice",
            "instructions": "What is the primary entity type featured in this passage?",
            "criteria": {
                "person": "A named individual, politician, or resident",
                "organization": "A business, railroad, institution, or government agency",
                "location": "A specific town, county, state, or geographic area",
                "general_text": "No single prominent entity or general commentary"
            }
        },
        "has_person": {
            "type": "noul",
            "instructions": "Does this excerpt mention a named individual or historical figure?"
        },
        "has_location": {
            "type": "noul",
            "instructions": "Does this excerpt mention a specific city, town, county, or state?"
        },
        "has_organization": {
            "type": "noul",
            "instructions": "Does this excerpt mention an institution, company, railroad, court, or committee?"
        },
        "editorial_type": {
            "type": "choice",
            "instructions": "What is the editorial purpose or format of this item?",
            "criteria": {
                "news_report": "Objective or factual reporting of events",
                "opinion_editorial": "Editorial commentary, opinion, or political persuasion",
                "commercial_ad": "Paid advertisement, business announcement, or product claim",
                "legal_notice": "Official notice, sheriff sale, summons, or ordinance"
            }
        },
        "historical_interest": {
            "type": "score",
            "instructions": "How significant is the historical news value of this piece?",
            "criteria": [
                "Minor routine notice or brief advertisement",
                "Local interest story or standard city report",
                "High historical, statewide, or national importance report"
            ]
        }
    },
    "ner": {
        "primary_entity": {
            "type": "choice",
            "instructions": "What is the primary entity type featured in this passage?",
            "criteria": {
                "person": "A named individual, politician, or resident",
                "organization": "A business, railroad, institution, or government agency",
                "location": "A specific town, county, state, or geographic area",
                "general_text": "No single prominent entity or general commentary"
            }
        },
        "has_person": {
            "type": "noul",
            "instructions": "Does this excerpt mention a named individual or historical figure?"
        },
        "has_location": {
            "type": "noul",
            "instructions": "Does this excerpt mention a specific city, town, county, or state?"
        },
        "has_organization": {
            "type": "noul",
            "instructions": "Does this excerpt mention an institution, company, railroad, court, or committee?"
        },
        "has_money_or_prices": {
            "type": "noul",
            "instructions": "Does this excerpt mention specific monetary values, prices, dollars, or financial sums?"
        }
    },
    "classification": {
        "topic": {
            "type": "choice",
            "instructions": "What is the primary topic of this newspaper excerpt?",
            "criteria": {
                "politics_government": "Elections, state affairs, legislation, and political leaders",
                "crime_courts": "Legal trials, crimes, police activity, and judicial verdicts",
                "business_markets": "Commodity prices, trade, railroad reports, and financial news",
                "society_local": "Local community updates, social notices, obituaries, and events",
                "advertisement": "Classified advertisements, product pitches, remedies, and notices"
            }
        },
        "is_crime_or_accident": {
            "type": "noul",
            "instructions": "Does this piece report a crime, accident, disaster, casualty, or death?"
        },
        "is_advertisement": {
            "type": "noul",
            "instructions": "Is this text a commercial advertisement or business promotion?"
        }
    },
    "triage": {
        "urgency": {
            "type": "score",
            "instructions": "How urgent was this announcement for contemporary readers?",
            "criteria": [
                "Routine or passive information",
                "Action or interest expected within days",
                "Immediate breaking crisis, emergency, or major deadline"
            ]
        },
        "sentiment": {
            "type": "choice",
            "instructions": "What is the predominant sentiment or tone?",
            "criteria": {
                "positive": "Celebratory, optimistic, successful, or laudatory",
                "negative": "Alarming, tragic, critical, adversarial, or distressed",
                "neutral": "Matter-of-fact, plain factual, or formal"
            }
        },
        "is_disaster_or_conflict": {
            "type": "noul",
            "instructions": "Does this text report war, violent conflict, severe fire, or natural disaster?"
        }
    },
    "ocr": {
        "text_quality": {
            "type": "choice",
            "instructions": "How would you classify the linguistic quality and legibility of this text?",
            "criteria": {
                "coherent_sentences": "Meaningful text containing recognizable English sentences, proper syntax, and vocabulary",
                "gibberish_noise": "Unintelligible character noise, broken symbols, or meaningless fragments"
            }
        },
        "is_severe_ocr_noise": {
            "type": "noul",
            "instructions": "Is this text dominated by OCR noise, broken fragments, and unreadable characters?"
        },
        "has_ocr_flaws": {
            "type": "noul",
            "instructions": "Does this text contain noticeable optical character recognition (OCR) scanning errors, typos, or broken words?"
        }
    }
}


COMMON_STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
}
VOWELS = set("aeiouyAEIOUY")
WEIRD_OCR_SYMBOLS = set("^~|_*\§#<>{}[]=@`")


def analyze_ocr_lexical(text: str) -> Dict[str, Any]:
    """Analyze character- and token-level lexical signals for OCR quality."""
    if not text or not text.strip():
        return {
            "score": 0.0,
            "verdict": "BAD_OCR",
            "alpha_ratio": 0.0,
            "symbol_noise": 0.0,
            "vowel_word_ratio": 0.0,
            "stopword_ratio": 0.0,
            "fragmentation": 0.0,
            "avg_word_len": 0.0,
            "total_words": 0,
            "flags": ["Empty or whitespace-only text"]
        }

    total_chars = len(text)
    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / total_chars

    weird_symbols = sum(1 for c in text if c in WEIRD_OCR_SYMBOLS)
    symbol_noise_ratio = weird_symbols / total_chars

    words = re.findall(r"[a-zA-Z]+", text)
    total_words = len(words)
    if total_words == 0:
        return {
            "score": 0.0,
            "verdict": "BAD_OCR",
            "alpha_ratio": round(alpha_ratio, 3),
            "symbol_noise": round(symbol_noise_ratio, 3),
            "vowel_word_ratio": 0.0,
            "stopword_ratio": 0.0,
            "fragmentation": 0.0,
            "avg_word_len": 0.0,
            "total_words": 0,
            "flags": ["No alphabetic words detected"]
        }

    words_lower = [w.lower() for w in words]
    stopword_count = sum(1 for w in words_lower if w in COMMON_STOPWORDS)
    stopword_ratio = stopword_count / total_words

    has_vowel_count = sum(1 for w in words if any(c in VOWELS for c in w))
    vowel_word_ratio = has_vowel_count / total_words

    single_letters = sum(1 for w in words_lower if len(w) == 1 and w not in ("a", "i"))
    fragmentation = single_letters / total_words

    avg_len = sum(len(w) for w in words) / total_words

    flags: List[str] = []
    penalty = 0.0

    if alpha_ratio < 0.60:
        penalty += 0.35
        flags.append(f"Excessive non-letters (only {alpha_ratio:.1%} alphabetic)")
    elif alpha_ratio < 0.72:
        penalty += 0.15
        flags.append(f"Low alphabetic character ratio ({alpha_ratio:.1%})")

    if symbol_noise_ratio > 0.03:
        penalty += 0.30
        flags.append(f"Heavy scanner symbol artifacts ({symbol_noise_ratio:.1%})")
    elif symbol_noise_ratio > 0.01:
        penalty += 0.15
        flags.append(f"Elevated scan noise artifacts ({symbol_noise_ratio:.1%})")

    if vowel_word_ratio < 0.80:
        penalty += 0.30
        flags.append(f"High proportion of vowelless fragments ({1 - vowel_word_ratio:.1%})")
    elif vowel_word_ratio < 0.90:
        penalty += 0.15
        flags.append(f"Noticeable vowelless tokens ({1 - vowel_word_ratio:.1%})")

    if stopword_ratio < 0.10:
        penalty += 0.25
        flags.append(f"Abnormally low stopword density ({stopword_ratio:.1%})")
    elif stopword_ratio < 0.18:
        penalty += 0.10
        flags.append(f"Low stopword density ({stopword_ratio:.1%})")

    if fragmentation > 0.10:
        penalty += 0.25
        flags.append(f"High word fragmentation / isolated letters ({fragmentation:.1%})")

    lex_score = max(0.0, min(1.0, 1.0 - penalty))

    return {
        "score": round(lex_score, 2),
        "alpha_ratio": round(alpha_ratio, 3),
        "symbol_noise": round(symbol_noise_ratio, 3),
        "vowel_word_ratio": round(vowel_word_ratio, 3),
        "stopword_ratio": round(stopword_ratio, 3),
        "fragmentation": round(fragmentation, 3),
        "avg_word_len": round(avg_len, 2),
        "total_words": total_words,
        "flags": flags
    }


def classify_ocr(text: str, laya_answers: Dict[str, Any]) -> Dict[str, Any]:
    """Combine Laya System 1 neural decisions with lexical surface metrics to classify OCR quality."""
    lex = analyze_ocr_lexical(text)

    noise = laya_answers.get("is_severe_ocr_noise", {}).get("noul", 0.0)
    flaws = laya_answers.get("has_ocr_flaws", {}).get("noul", 0.0)
    tq_data = laya_answers.get("text_quality", {})
    tq_choice = tq_data.get("choice", "coherent_sentences")
    tq_gibberish_prob = tq_data.get("probabilities", {}).get("gibberish_noise", 0.0)

    # Laya semantic health index (0.0 to 1.0)
    laya_health = max(0.0, 1.0 - (noise * 0.6 + flaws * 0.4))
    if tq_choice == "gibberish_noise":
        laya_health = min(laya_health, 1.0 - tq_gibberish_prob)

    composite_score = int(round((laya_health * 0.5 + lex["score"] * 0.5) * 100))

    reasons = list(lex["flags"])
    if noise >= 0.40:
        reasons.append(f"Laya neural head flagged severe OCR distortion (prob: {noise:.2f})")
    elif flaws >= 0.40:
        reasons.append(f"Laya neural head flagged frequent OCR typos/flaws (prob: {flaws:.2f})")
    if tq_choice == "gibberish_noise" and tq_gibberish_prob >= 0.50:
        reasons.append(f"Laya classified text structure as gibberish noise ({tq_gibberish_prob:.1%})")

    if composite_score >= 70 and noise < 0.35 and lex["score"] >= 0.65:
        verdict = "GOOD_OCR"
    elif composite_score <= 45 or noise >= 0.55 or lex["score"] <= 0.35 or tq_gibberish_prob >= 0.70:
        verdict = "BAD_OCR"
    else:
        verdict = "MARGINAL_OCR"

    return {
        "verdict": verdict,
        "score": composite_score,
        "laya": {
            "severe_noise_prob": round(noise, 4),
            "flaws_prob": round(flaws, 4),
            "text_quality": tq_choice,
            "gibberish_prob": round(tq_gibberish_prob, 4),
            "laya_health_pct": int(round(laya_health * 100))
        },
        "lexical": lex,
        "reasons": reasons
    }


def make_bar(prob: float, width: int = 18) -> str:
    """Render a visual probability bar."""
    filled = int(round(prob * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def fetch_random_pinecone_document(
    index_name: str,
    api_key: Optional[str] = None,
    namespace: Optional[str] = None,
    doc_id: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch a document from a Pinecone Document Schema index."""
    if Pinecone is None:
        raise RuntimeError("pinecone package not found. Run: pip install pinecone")

    pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
    idx = pc.Index(index_name)

    # 1. Determine namespace
    if not namespace:
        stats = idx.describe_index_stats()
        available_namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        if not available_namespaces:
            raise ValueError(f"No namespaces found in index {index_name}")
        namespace = random.choice(available_namespaces)

    # 2. Determine document ID
    if not doc_id:
        # Note: Avoid list(idx.documents.list(...)) directly as the Paginator
        # will recursively fetch every page across the entire namespace.
        docs_paginator = idx.documents.list(namespace=namespace, limit=50)
        doc_records = list(itertools.islice(docs_paginator, 50))
        if not doc_records:
            raise ValueError(f"No documents found in namespace '{namespace}'")
        selected_record = random.choice(doc_records)
        doc_id = selected_record.id if hasattr(selected_record, "id") else str(selected_record)

    # 3. Fetch full document content
    fetched = idx.documents.fetch(namespace=namespace, ids=[doc_id])
    if not fetched.documents or doc_id not in fetched.documents:
        raise KeyError(f"Document ID '{doc_id}' not found in namespace '{namespace}'")

    doc = fetched.documents[doc_id]
    doc_dict = doc.to_dict() if hasattr(doc, "to_dict") else dict(doc)
    doc_dict["_namespace"] = namespace
    doc_dict["_id"] = doc_id
    return doc_dict


def run_laya_test(
    text: str,
    questions: Dict[str, Any],
    router: Any,
    model_override: Optional[str] = None,
    max_len: int = 1024
) -> Dict[str, Any]:
    """Run Laya System 1 decision inference."""
    kwargs: Dict[str, Any] = {}
    if model_override:
        kwargs["model"] = model_override
    if max_len:
        kwargs["max_len"] = max_len

    return router.predict(text, questions, **kwargs)


def display_results(
    text: str,
    metadata: Dict[str, Any],
    result: Dict[str, Any],
    preset: str = "full",
    max_preview_chars: int = 400
) -> None:
    """Print beautifully formatted Laya decision results to terminal."""
    print("\n" + "═" * 74)
    print("  📰 SOURCE DOCUMENT METADATA")
    print("═" * 74)
    if "_id" in metadata:
        print(f"  • Document ID : {metadata.get('_id')}")
        print(f"  • Namespace   : {metadata.get('_namespace')}")
    if "newspaper_title" in metadata:
        print(f"  • Newspaper   : {metadata.get('newspaper_title')} ({metadata.get('date', 'Unknown')})")
        print(f"  • Location    : {metadata.get('city', 'Unknown')}, {metadata.get('state', 'Unknown')}")
    if "loc_page_url" in metadata:
        print(f"  • Chronicling : {metadata.get('loc_page_url')}")
    print(f"  • Text Length : {len(text):,} characters")

    print("\n" + "─" * 74)
    print("  📜 INPUT TEXT PREVIEW (First chunk passed to model)")
    print("─" * 74)
    clean_preview = text[:max_preview_chars].strip()
    for line in clean_preview.splitlines():
        if line.strip():
            print(f"    {line}")
    if len(text) > max_preview_chars:
        print(f"    ... [{len(text) - max_preview_chars:,} more characters]")

    answers = result.get("answers", {})

    # If OCR preset is selected or OCR questions are present, display the OCR Audit Card
    if preset == "ocr" or "is_severe_ocr_noise" in answers:
        qc = classify_ocr(text, answers)
        verdict = qc["verdict"]
        badge = {
            "GOOD_OCR": "\033[1;42;30m GOOD OCR \033[0m \033[1;32m(Clean & Highly Legible)\033[0m",
            "MARGINAL_OCR": "\033[1;43;30m MARGINAL OCR \033[0m \033[1;33m(Readable with scattered typos)\033[0m",
            "BAD_OCR": "\033[1;41;37m BAD OCR \033[0m \033[1;31m(Severe noise or gibberish)\033[0m"
        }.get(verdict, verdict)

        lex = qc["lexical"]
        laya_info = qc["laya"]

        print("\n" + "═" * 74)
        print("  🔍 OCR QUALITY AUDIT REPORT")
        print("═" * 74)
        print(f"  • Final Assessment   : {badge}")
        print(f"  • Overall OCR Score  : \033[1m{qc['score']} / 100\033[0m  [{make_bar(qc['score'] / 100, 16)}]")
        print(f"  • Laya Health Index  : {laya_info['laya_health_pct']}%  [{make_bar(laya_info['laya_health_pct'] / 100, 16)}]")
        print(f"  • Lexical Integrity  : {int(lex['score'] * 100)}%  [{make_bar(lex['score'], 16)}]")

        print("\n  📊 Surface Lexical Metrics:")
        print(f"    • Alphabetic Ratio    : {lex['alpha_ratio']:.1%} of characters")
        print(f"    • Symbol / Scan Noise : {lex['symbol_noise']:.1%}")
        print(f"    • Vowel-Formed Words  : {lex['vowel_word_ratio']:.1%}")
        print(f"    • Common Stopwords    : {lex['stopword_ratio']:.1%}")
        print(f"    • Word Fragmentation  : {lex['fragmentation']:.1%} (isolated single letters)")
        print(f"    • Avg Word Length     : {lex['avg_word_len']} chars ({lex['total_words']} words detected)")

        print("\n  ⚡ Laya Neural Signals:")
        print(f"    • Severe Distortion Prob : {laya_info['severe_noise_prob']:.4f}  [{make_bar(laya_info['severe_noise_prob'], 16)}]")
        print(f"    • Scanning Flaws Prob    : {laya_info['flaws_prob']:.4f}  [{make_bar(laya_info['flaws_prob'], 16)}]")
        print(f"    • Linguistic Structure   : \033[1;36m{laya_info['text_quality']}\033[0m")

        if qc["reasons"]:
            print("\n  📝 Diagnostic Flags:")
            for flag in qc["reasons"]:
                print(f"    ⚠️  {flag}")
        else:
            print("\n  📝 Diagnostic Flags:")
            print("    ✅ No major optical degradation or character anomalies detected.")

    print("\n" + "═" * 74)
    print("  ⚡ LAYA SYSTEM 1 DECISION RESULTS")
    print("═" * 74)
    for q_name, ans in answers.items():
        ans_type = ans.get("type")
        confidence = ans.get("confidence", 0.0)

        if ans_type == "choice":
            chosen = ans.get("choice")
            print(f"\n  🏷️  [{q_name.upper()}] (choice) → Selected: \033[1;36m{chosen}\033[0m")
            print(f"      Confidence: {confidence:.2f}")
            print("      Probabilities:")
            for opt, prob in ans.get("probabilities", {}).items():
                bar = make_bar(prob, 16)
                marker = " ◄" if opt == chosen else ""
                print(f"        • {opt:<24} : {prob:.4f}  [{bar}]{marker}")

        elif ans_type == "noul":
            prob = ans.get("noul", 0.0)
            bar = make_bar(prob, 16)
            verdict = "\033[1;32mYES\033[0m" if prob >= 0.5 else "\033[1;31mNO\033[0m"
            print(f"\n  🔘 [{q_name.upper()}] (yes/no) → {verdict} (Prob: \033[1m{prob:.4f}\033[0m | Conf: {confidence:.2f})")
            print(f"      [{bar}]")

        elif ans_type == "score":
            score = ans.get("score", 0.0)
            legend = ans.get("legend", {})
            max_lvl = len(legend) - 1 if legend else 1
            print(f"\n  📊 [{q_name.upper()}] (score) → Expected Level: \033[1;33m{score:.2f} / {max_lvl}\033[0m")
            print(f"      Confidence: {confidence:.2f}")
            print("      Level Distribution:")
            for lvl, prob in ans.get("probabilities", {}).items():
                desc = legend.get(lvl, f"Level {lvl}")
                bar = make_bar(prob, 16)
                print(f"        • Lvl {lvl} ({desc:<40}) : {prob:.4f}  [{bar}]")

    routing = result.get("routing", {})
    usage = result.get("usage", {})
    print("\n" + "─" * 74)
    print("  ⚙️  INFERENCE METRICS")
    print("─" * 74)
    print(f"  • Model Checkpoint : {routing.get('model')} ({routing.get('repo')})")
    print(f"  • Routing Reason   : {routing.get('reason')}")
    lang_info = routing.get("detection", {})
    print(f"  • Language/Script  : {lang_info.get('language')} / {lang_info.get('script')}")
    print(f"  • Input Tokens     : {usage.get('input_tokens')}")
    print("═" * 74 + "\n")


def evaluate_ocr_quality(
    text: str,
    router: Optional[Any] = None,
    max_chars: int = 900
) -> Dict[str, Any]:
    """Classify text chunk as good, marginal, or bad OCR using Laya + lexical analysis."""
    if router is None:
        if Router is None:
            raise RuntimeError("laya package not found. Run: pip install laya")
        router = Router()
    sample = text[:max_chars]
    questions = QUESTION_PRESETS["ocr"]
    result = router.predict(sample, questions)
    return classify_ocr(sample, result.get("answers", {}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Laya decision engine on random or specific Pinecone chunks."
    )
    parser.add_argument(
        "--index",
        default=os.getenv("PINECONE_INDEX"),
        help="Pinecone index name (default: from PINECONE_INDEX env var)"
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Pinecone namespace/year (e.g. '1906', '1891'). Default: random namespace."
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Exact Pinecone document ID to test."
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Custom text string to test directly (bypasses Pinecone)."
    )
    parser.add_argument(
        "--preset",
        choices=list(QUESTION_PRESETS.keys()),
        default="full",
        help="Question preset: 'full', 'ner', 'classification', 'triage', or 'ocr' (default: full)."
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=900,
        help="Maximum characters of chunk text to send to Laya (default: 900)."
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=["english", "multilingual", "typed-decisions"],
        help="Override checkpoint selection (default: automatic routing)."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running in an interactive loop fetching random chunks on Enter."
    )
    args = parser.parse_args()

    if Router is None:
        print("Error: 'laya' package is not installed.")
        print("Please install with: pip install laya")
        sys.exit(1)

    print("Initializing Laya Router...")
    router = Router()
    questions = QUESTION_PRESETS[args.preset]

    while True:
        metadata: Dict[str, Any] = {}
        if args.text:
            text = args.text
            metadata = {"newspaper_title": "Manual CLI Input", "date": "N/A"}
        else:
            if not args.index:
                print("Error: No Pinecone index specified. Provide --index or set PINECONE_INDEX in .env.")
                sys.exit(1)

            print(f"Fetching chunk from Pinecone index '{args.index}'...")
            try:
                metadata = fetch_random_pinecone_document(
                    index_name=args.index,
                    namespace=args.namespace,
                    doc_id=args.doc_id
                )
                text = metadata.get("text", "")
                if not text:
                    print(f"Warning: Document {metadata.get('_id')} has no text field. Retrying...")
                    if not args.loop:
                        break
                    continue
            except Exception as e:
                print(f"Error fetching from Pinecone: {e}")
                sys.exit(1)

        input_slice = text[:args.max_chars]
        print(f"Evaluating with Laya [Preset: {args.preset}]...")
        result = run_laya_test(
            text=input_slice,
            questions=questions,
            router=router,
            model_override=args.model
        )

        display_results(text, metadata, result, preset=args.preset)

        if not args.loop:
            break

        try:
            user_input = input("Press [Enter] for next random chunk, or 'q' to quit: ")
            if user_input.strip().lower() in ("q", "quit", "exit"):
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
