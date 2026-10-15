"""Splits the model's Chain-of-Thought output into its two labeled sections.

Kept free of heavy imports so the UI can use it without loading the RAG stack.
"""
import re

# A section header at the start of a line, e.g. "**Answer:**", "Answer:",
# "2. Answer:", "### Answer", "**Answer**" or '"Answer:"'. A colon (or the
# line ending right after the label) is required, so prose such as
# "Answering this needs..." is never mistaken for a header.
_HEADER = (
    r"^[ \t]*(?:#{{1,6}}[ \t]*)?(?:\d+\.[ \t]*)?(?:\*\*|__)?[ \t]*[\"“]?{label}\b[\"”]?[ \t]*"
    r"(?::[ \t]*[\"”]?[ \t]*(?:\*\*|__)?|(?:\*\*|__)[ \t]*:|(?:\*\*|__)?[ \t]*$)[ \t]*"
)

ANSWER_RE = re.compile(_HEADER.format(label=r"(?:final[ \t]+)?answer"), re.IGNORECASE | re.MULTILINE)
REASONING_RE = re.compile(_HEADER.format(label=r"(?:retrieval[ \t]+)?reasoning"), re.IGNORECASE | re.MULTILINE)


def split_reasoning(text: str):
    """Returns (reasoning, answer, found_answer_header).

    If the "Answer:" header hasn't appeared (yet, while streaming, or at all),
    everything seen so far is treated as reasoning and answer is "".
    """
    text = text or ""
    match = ANSWER_RE.search(text)
    if not match:
        return REASONING_RE.sub("", text, count=1).strip(), "", False
    reasoning = REASONING_RE.sub("", text[:match.start()], count=1).strip()
    return reasoning, text[match.end():].strip(), True


def final_sections(text: str):
    """Like split_reasoning, but for a finished response: if the model never
    wrote an "Answer:" header, the whole text is the answer."""
    reasoning, answer, found = split_reasoning(text)
    if not found:
        return "", (text or "").strip()
    return reasoning, answer
