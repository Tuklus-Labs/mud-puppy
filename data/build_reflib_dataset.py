"""Generate a TinyLlama QLoRA training set from the AEGIS reference library.

Walks every .md file under ~/Projects/Aegis/AEGIS/docs/reference-library/,
parses markdown headings, and emits JSONL chat-format training examples.

Output format (matches tokenizer.apply_chat_template for TinyLlama-Chat):

    {"messages": [
        {"role": "user", "content": "Explain the Free Energy Principle."},
        {"role": "assistant", "content": "The Free Energy Principle is ..."}
    ]}

Strategy:
    - Each ## or ### heading becomes a Q/A pair where Q is synthesized from
      the heading text and A is the section body up to the next heading.
    - Top-of-doc (# H1) sections become a "What is <topic>?" pair using the
      first paragraph as the answer.
    - Short sections (<80 chars) and huge sections (>3500 chars) are filtered.
    - ~4-8 templates per heading for question phrasing diversity.
    - Blockquotes become "What does <cited> say about <topic>?" pairs when
      a citation line is present.

No LLM synthesis, pure mechanical extraction. Good enough for a SFT pass.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

REFLIB = Path.home() / "Projects/Aegis/AEGIS/docs/reference-library"
OUT = Path.home() / "Projects/mud-puppy/data/aegis-reflib-qlora.jsonl"
SEED = 1337

# Sections smaller than this in characters are skipped (too little signal).
MIN_LEN = 80
# Sections larger than this are truncated; the tokenizer will clip further.
MAX_LEN = 3500

# Question phrasing templates.  Pick one at random per section for variety.
Q_TEMPLATES = [
    "Explain {t}.",
    "What is {t}?",
    "Tell me about {t}.",
    "Give a concise explanation of {t}.",
    "Can you walk me through {t}?",
    "Summarize {t}.",
    "What does {t} mean?",
    "Describe {t} in detail.",
]

# Templates used when the heading looks like a full phrase (already a question
# or instruction-like), so we use it more directly.
Q_DIRECT_TEMPLATES = [
    "{t}",
    "Please elaborate: {t}",
    "Can you clarify: {t}",
]

# Match a markdown heading line: "#", "##", "###" followed by text.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Strip leading numeric tokens from headings: "1.2 Variational Free Energy"
# -> "Variational Free Energy"; "03-spreading-activation" -> "spreading-activation".
STRIP_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*\s+)")

# Match "> text" continuation blocks; consecutive > lines form a quote.
QUOTE_RE = re.compile(r"^>\s?(.*)$")

# Citation line inside a quote: "-- Author" or "— Author".
CITE_RE = re.compile(r"^\s*[-—]{1,2}\s*(.+?)\s*$")


def clean_heading(raw: str) -> str:
    """Strip leading section numbers and junk; title-case acronyms stay.

    Handles: "1.2 Foo" -> "Foo", "-- Foo --" -> "Foo",
    "Doc 232: Foo" -> "Foo", trailing punctuation stripped.
    """
    t = raw.strip()
    # Strip leading "--" / em-dashes / colons.
    t = re.sub(r"^[-—:\s]+", "", t)
    # Strip trailing "--" and dashes/colons.
    t = re.sub(r"[-—:\s]+$", "", t)
    # Strip leading section numbers: "1.2.3 Foo" -> "Foo".
    t = STRIP_NUMBER_RE.sub("", t)
    # Strip "Doc NNN:" / "Doc NNN --" prefixes.
    t = re.sub(r"^Doc\s+\d+\s*[:\-—]?\s*", "", t, flags=re.IGNORECASE)
    # Collapse internal repeated whitespace.
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def pick_question(heading: str, rng: random.Random) -> str:
    """Synthesize a question from a heading."""
    # If heading already looks like a question, use it directly.
    if heading.endswith("?"):
        return heading
    # Short keyword headings get template-wrapped.
    tpl_set = Q_TEMPLATES if len(heading) < 80 else Q_DIRECT_TEMPLATES
    return rng.choice(tpl_set).format(t=heading)


def parse_sections(md: str) -> list[tuple[int, str, str]]:
    """Split a markdown doc into (level, heading, body) tuples.

    Level is the number of # characters. Body includes everything after the
    heading up to the next heading of equal-or-greater level.
    """
    lines = md.splitlines()
    out: list[tuple[int, str, str]] = []
    cur_level = 0
    cur_heading = ""
    cur_body: list[str] = []

    def flush() -> None:
        if cur_heading:
            body = "\n".join(cur_body).strip()
            if body:
                out.append((cur_level, cur_heading, body))

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            cur_level = len(m.group(1))
            cur_heading = clean_heading(m.group(2))
            cur_body = []
        else:
            cur_body.append(line)

    flush()
    return out


def extract_quotes(md: str) -> list[tuple[str, str]]:
    """Return (quote_text, citation) pairs from markdown blockquote blocks."""
    lines = md.splitlines()
    pairs: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        # Look for a trailing citation like "-- Author" in the last line.
        cite = ""
        if buf and CITE_RE.match(buf[-1].strip()):
            cite_m = CITE_RE.match(buf[-1].strip())
            cite = cite_m.group(1) if cite_m else ""
            body = "\n".join(buf[:-1]).strip()
        else:
            body = "\n".join(buf).strip()
        if body and cite and len(body) > 40:
            pairs.append((body, cite))

    for line in lines:
        qm = QUOTE_RE.match(line)
        if qm:
            buf.append(qm.group(1))
        else:
            flush()
            buf = []
    flush()
    return pairs


def doc_topic(path: Path) -> str:
    """Extract a human topic name from a doc path / first heading."""
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            return clean_heading(m.group(2))
    # Fall back to filename: "01-active-inference.md" -> "active inference"
    stem = path.stem
    stem = STRIP_NUMBER_RE.sub("", stem + " ").strip()
    stem = stem.replace("-", " ").strip()
    return stem or path.stem


def build_pairs_from_doc(path: Path, rng: random.Random) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    topic = doc_topic(path)
    sections = parse_sections(text)
    pairs: list[dict] = []

    # 1. Doc-level "what is topic?" using first body paragraph.
    if sections:
        first_lvl, first_head, first_body = sections[0]
        # Use first non-quote paragraph.
        paragraphs = [p for p in first_body.split("\n\n") if p.strip() and not p.startswith(">")]
        if paragraphs:
            first_para = paragraphs[0].strip()
            if MIN_LEN <= len(first_para) <= MAX_LEN:
                pairs.append(
                    {
                        "messages": [
                            {"role": "user", "content": f"What is {topic}?"},
                            {"role": "assistant", "content": first_para},
                        ]
                    }
                )

    # 2. Per-section Q/A.
    for level, heading, body in sections:
        # Skip H1 (already used above) and very shallow sub-subsubheadings.
        if level < 2 or level > 4:
            continue
        body = body.strip()
        if len(body) < MIN_LEN:
            continue
        body = body[:MAX_LEN]
        # Reject responses that are mostly non-prose: very few word characters
        # vs total length often means we extracted a half-broken code block.
        word_chars = sum(1 for c in body if c.isalpha())
        if word_chars < max(40, len(body) * 0.25):
            continue
        # Reject responses that are mostly triple-backtick code-fence scaffolding
        # with no actual content.
        if body.count("```") >= 2 and len(re.sub(r"```[^`]*```", "", body).strip()) < 40:
            continue
        # Skip sections that are just a list of links or too dense with symbols.
        if body.count("\n") > 0 and all(l.strip().startswith(("-", "*")) for l in body.splitlines() if l.strip()):
            # Pure bullet list — still useful but less so; keep with lower prob.
            if rng.random() > 0.4:
                continue
        # Heading sanity: skip headings that survived cleaning with too little
        # signal (e.g. "", "--", stray punctuation).
        if len(heading) < 3 or not any(c.isalpha() for c in heading):
            continue
        q = pick_question(heading, rng)
        pairs.append(
            {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": body},
                ]
            }
        )

    # 3. Quote-extraction pairs.
    for quote, cite in extract_quotes(text):
        q = f"What does {cite} say about {topic}?"
        a = f'"{quote.strip()}"'
        if MIN_LEN <= len(a) <= MAX_LEN:
            pairs.append(
                {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
            )

    return pairs


def main() -> None:
    rng = random.Random(SEED)
    all_pairs: list[dict] = []
    docs = sorted(REFLIB.glob("*.md"))
    print(f"[dataset] scanning {len(docs)} docs in {REFLIB}")

    for path in docs:
        try:
            pairs = build_pairs_from_doc(path, rng)
            all_pairs.extend(pairs)
        except Exception as exc:
            print(f"[dataset] skipped {path.name}: {exc}")

    # Shuffle so consecutive rows don't come from the same doc (helps train).
    rng.shuffle(all_pairs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for ex in all_pairs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Size estimate.
    size_bytes = OUT.stat().st_size
    print(f"[dataset] wrote {len(all_pairs):,} examples -> {OUT}")
    print(f"[dataset] file size: {size_bytes / 1e6:.2f} MB")

    # Quick token estimate assuming ~4 chars/token.
    total_chars = sum(
        len(ex["messages"][0]["content"]) + len(ex["messages"][1]["content"])
        for ex in all_pairs
    )
    est_tokens = total_chars // 4
    print(f"[dataset] ~{est_tokens:,} tokens (rough estimate)")

    # Per-section length distribution.
    lens = [
        len(ex["messages"][1]["content"])
        for ex in all_pairs
    ]
    if lens:
        lens.sort()
        print(
            f"[dataset] response length: min={lens[0]} p50={lens[len(lens)//2]} "
            f"p95={lens[int(len(lens)*0.95)]} max={lens[-1]}"
        )


if __name__ == "__main__":
    main()
