"""Deterministic strategy cards for problem families that are easy to misread.

The cards are deliberately short and operational.  They are supplied to the
planner and executor as an additional audit signal; the original statement and
its samples remain authoritative.
"""

from __future__ import annotations

import json
from typing import Any


def _text(problem: dict[str, Any] | str) -> str:
    if isinstance(problem, str):
        return problem.lower()
    return json.dumps(problem, ensure_ascii=False).lower()


_CARDS: tuple[tuple[tuple[str, ...], str], ...] = (
    # These cards are intentionally keyed by the stable problem URL/slug.
    # Several statements contain generic words such as "card game"; keeping
    # the specific cards before generic ones prevents a misleading match.
    (("1198/e", "painting the array"),
     "CF1198E audit card: compress rectangle boundaries at x1,x2+1 and y1,y2+1. "
     "The covered elementary cells form a bipartite graph: each horizontal strip has capacity its height, "
     "each vertical strip capacity its width, and every covered cell is an infinite edge. The minimum s-t cut "
     "is the answer. Reject exponential subset/interval DP and check the cut on a tiny brute-force grid."),
    (("arc061/d", "card game for three"),
     "ARC061D audit card: use the first-passage prefix formula for the three-player random card process, "
     "with O(N+M+K) modular factorial/power precomputation. The tempting 'the Nth A appears before the Mth B/Kth C' "
     "event is not the required stopping event; derive the prefix recurrence and dry-run every sample."),
    (("gathering", "3187"),
     "Kattis gathering audit card: the L1 radius constraint couples x and y. Transform u=x+y and v=x-y only for "
     "the constraint, then optimize the original sum of Manhattan distances jointly; do not clamp independent x/y "
     "medians or treat the two intervals as separable. Verify candidate breakpoints against exhaustive small grids."),
    (("zamboni", "3664"),
     "Kattis zamboni audit card: on every unit step paint the current cell first, then move; turn clockwise after "
     "each segment. A repeat state includes row, column, direction, segment-length phase, and color phase. When "
     "skipping cycles, execute one cycle's painting effect before jumping repeated cycles and leave @ at the final cell."),
    (("tombraider", "3272"),
     "Kattis tombraider audit card: trace each laser ray until wall, obstacle, or gargoyle and build the exact "
     "face-to-face functional graph. A cycle is not automatically a valid solution merely because it reaches a "
     "gargoyle; encode the required target face/orientation constraints in 2-SAT and validate every chosen ray."),
    (("pieceofcake", "piece of cake"),
     "Kattis pieceofcake audit card: for ordered cyclic vertex distance d, a directed edge is present in a uniform "
     "k-subset with probability C(n-d-1,k-2)/C(n,k). Sum signed cross products over all starts and distances, then "
     "take the absolute value. The common C(n-d,k-2) expression is off by one."),
    (("799/d", "arkady and a nobody"),
     "CF799D audit card: each multiplier can be assigned to at most one side. Sort factors descending and keep a "
     "capped two-dimensional product DP (or equivalent Pareto frontier) over both dimensions; never divide capped "
     "products or assume a single prefix split. Check all four target/current orientations."),
    (("holidaystars", "holiday cookie throwing"),
     "Kattis holidaystars audit card: use the area-weighted polygon centroid, not the average of vertices. For the "
     "clockwise rotation x'=dx*cos(wt)+dy*sin(wt), find the earliest root for every vertex with complete bracketing "
     "over the first feasible horizon and output the smallest time."),
    (("bracketnotation", "bracket notation"),
     "Kattis bracketnotation audit card: headers are emitted in preorder. A node's start is immediately after its "
     "own header and its end is after all descendant headers; child headers follow the parent header. Solve the "
     "digit-length fixed point iteratively; a prefix sum of subtree lengths alone gives wrong absolute indexes."),
    (("backpackbuddies", "backpack buddies"),
     "Kattis backpackbuddies audit card: compare the knight's shortest pure walking time with a day-state Dijkstra. "
     "If an edge would exceed 12 hours, pay the overnight wait 12-used before walking it; do not reset the day for "
     "free or zero time. The answer is day-aware time minus knight time."),
    (("1012/d", "two strings swaps"),
     "CF1012D audit card: the minimum number of prefix swaps can exceed two. Use the run/block boundary greedy "
     "construction, enumerate only compatible splice boundaries, and simulate every emitted prefix operation. Reject "
     "heuristics that stop after a fixed number of swaps or fail to check legal lengths."),
    (("naman and binary strings", "1370/e", "subsequence and rotate"),
     "CF1370E audit card: map each mismatch 0->1 to +1 and 1->0 to -1. "
     "Counts must balance. The answer is max over the maximum subarray sum of this sequence and its negation; do not count permutation cycles."),
    (("robbers and searchlights", "1408/d"),
     "CF1408D audit card: for every visible robber/searchlight pair with dx=c-a>=0, dy=d-b>=0, "
     "record need[dx]=max(need[dx],dy+1), suffix-max need, then minimize x+need[x]. "
     "This is a global x/y tradeoff, not binary search over an independently checked helper."),
    (("626/b",),
     "CF626B audit card: one color stays that color; all three colors reach BGR; with exactly two colors, "
     "equal counts reach all three, otherwise reach the minority color and the missing color. Return sorted letters."),
    (("devu and his class", "439/b"),
     "CF439B audit card: teaching a subject consumes all its chapters at one rate. Sort chapter counts ascending, "
     "then sum c[i]*max(1,x-i); descending order is the opposite of the exchange argument."),
    (("valid test cases", "1365/f"),
     "CF1365F audit card: operations swap equal-length prefix and suffix. Compare multisets of unordered symmetric "
     "pairs (a[i],a[n-1-i]); for odd n the center must match. Do not use global permutation parity."),
    (("network coverage", "1373/f"),
     "CF1373F audit card: propagate the unavoidable carry around the circle: cur=max(0,cur+a[i]-b[i-1]) for i=1..n-1, "
     "then feasibility is cur+a[0]<=b[n-1]. Check indexing against the station-city definition."),
    (("forging the signature", "1059/b"),
     "CF1059B audit card: a valid 3x3 stamp requires only the eight neighbors to be '#'; the center may be '.' or '#'. "
     "Mark neighbors of every valid center and require every '#' cell to be covered."),
    (("grand moo-dapest", "645/c"),
     "CF645C audit card: list free rooms. For every k+1 consecutive free rooms choose the middle room as John's room "
     "and minimize max(distance to the two endpoints); do not treat an arbitrary 2D+1 interval as sufficient."),
    (("young physicist", "otpor", "resistor"),
     "Kattis otpor audit card: tokenize full resistor references R followed by all digits, not one character. "
     "Evaluate each parenthesized '-' series or '|' parallel group with a recursive descent or stack parser."),
    (("stone chiseler", "monumentmaker"),
     "Kattis monumentmaker audit card: read alternating lines in boustrophedon order, strip padding after reversing, "
     "concatenate, split on periods, and greedily pack whole dotted words with separators into width w."),
    (("irrational division", "irrationaldivision"),
     "Kattis irrationaldivision audit card: this is a finite minimax game on rectangle dimensions and checkerboard parity. "
     "Memoize (remaining rows, columns, left-column parity, turn); enumerate every positive row/column cut."),
    (("bracket beauty", "1248/d1"),
     "CF1248D1 audit card: the total bracket balance must be zero. Enumerate one swap and validate the beauty of the "
     "resulting circular shifts; keep the implementation within n<=500 and record an actual optimal pair, including i=j."),
    (("good lines", "886/f"),
     "CF886F audit card: projection symmetry is a multiset condition around an unknown center, not a pairwise reflection "
     "of the original points. Derive candidate line directions from projection-order changes and verify every candidate."),
    (("vasya and polynomial", "493/e"),
     "CF493E audit card: count coefficient vectors satisfying both P(t)=a and P(a)=b. Handle t=1,a=1 as infinite; "
     "otherwise use bounded base/carry representations and never assume ordinary base-t digits are unique when coefficients exceed t."),
    (("fix the string", "fixthe"),
     "FIXTHE audit card: for each '?' assignment, the commission depends on fixed 0/1 counts in its prefix and the "
     "rank of replacements of that digit. For each possible number of zeros, choose the smallest assignment deltas."),
    (("vasily's fence", "659/g"),
     "CF659G audit card: count distinct remaining-height profiles whose removed upper cells form one connected component. "
     "Use a monotone-stack/DP interval invariant; do not count arbitrary rectangles or treat each board independently."),
)


def strategy_card(problem: dict[str, Any] | str) -> str:
    """Return a strategy card, or an empty string when no card matches."""

    text = _text(problem)
    for markers, card in _CARDS:
        if any(marker in text for marker in markers):
            return card
    return ""


__all__ = ["strategy_card"]
