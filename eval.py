"""Router regression eval. Runs route_node only (no DB writes), compares
predicted node sets against expectations. Run before/after any change to
NODE_CATALOG, SYSTEM_PROMPT, or the model.

Each case: (text, must_match, may_also_match). Pass condition:
must_match <= predicted <= must_match | may_also_match
-- so genuinely ambiguous captures (task vs reminder) don't flake.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from graph import route_node  # noqa: E402  (must load .env first)

CASES = [
    # expense
    ("spent 40 on groceries", {"expense"}, set()),
    ("paid 1200 rent for july", {"expense"}, set()),
    ("coffee 4.50 at starbucks", {"expense"}, set()),
    # task
    ("buy milk and eggs", {"task"}, {"reminder"}),
    ("fix the squeaky bike brake", {"task"}, set()),
    ("clean out the garage sometime", {"task"}, set()),
    # reminder
    ("call mom tomorrow at 6pm", {"reminder"}, {"task"}),
    ("pick up dry cleaning near the mall", {"reminder"}, {"task"}),
    ("remind me to take my medicine at 9am", {"reminder"}, {"task"}),
    # idea
    ("app idea: AI recipe generator from fridge photos", {"idea"}, set()),
    ("what if the daily digest ranked items by urgency instead of time", {"idea"}, set()),
    # research
    ("https://simonwillison.net/2026/some-post-on-embeddings/ good breakdown", {"research"}, {"book", "idea"}),
    ("keep an eye on sqlite-vec releases for windows support", {"research"}, set()),
    # book
    ("started reading Atomic Habits, on chapter 3", {"book"}, set()),
    ("want to read Thinking Fast and Slow", {"book"}, set()),
    # multi-label
    ("bought Atomic Habits for 15 bucks, excited to read it", {"expense", "book"}, set()),
    ("submit the tax documents by friday evening", {"task", "reminder"}, set()),
    # no-fit -> empty matches (fallback inbox)
    ("asdkjhqwe zxcv mmmm", set(), set()),
]


def run_case(case):
    text, must, may = case
    result = route_node({"raw_text": text})
    predicted = {m["node"] for m in result["matches"]}
    confidences = {m["node"]: m["confidence"] for m in result["matches"]}
    passed = must <= predicted <= (must | may)
    return text, must, predicted, confidences, passed


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_case, CASES))

    failures = 0
    for text, must, predicted, confidences, passed in results:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        conf_str = ", ".join(f"{n}:{c:.2f}" for n, c in confidences.items()) or "-"
        print(f"[{mark}] {text!r}")
        print(f"       expected {sorted(must) or '[]'}  got {sorted(predicted) or '[]'}  ({conf_str})")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
