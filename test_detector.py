"""
Test Script for fake_news_detector.py v2.1
-------------------------------------------
Loads ~/fake_news_dataset.json, takes a stratified random 20% holdout,
fits the OnlineNormalizer on the 80% training split, builds a KD-Tree,
runs every test article through the weighted-KNN classifier, and prints
a full accuracy report with confusion matrix and confidence distribution.

Usage:
    python3 ~/test_detector.py

Requirements:
    - ~/fake_news_dataset.json  (created by setup_and_train.py)
    - Python 3.8+, no external libraries
"""

import json
import math
import os
import random
import re
import string
import sys
import time

# ── Colors ────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"; BOLD   = "\033[1m"
RED    = "\033[91m"; GREEN  = "\033[92m"
YELLOW = "\033[93m"; BLUE   = "\033[94m"
PURPLE = "\033[95m"; CYAN   = "\033[96m"
GRAY   = "\033[90m"

def c(t, col): return f"{col}{t}{RESET}"
def sep(ch="─", n=62): print(c(ch * n, GRAY))
def dsep(): print(c("═" * 62, CYAN))

DATASET_FILE = os.path.expanduser("~/fake_news_dataset.json")

# ─────────────────────────────────────────────────────────────────────────────
# EXACT COPIES of detector v2.0 internals
# ─────────────────────────────────────────────────────────────────────────────

SENSATIONAL_WORDS = [
    "shocking", "miracle", "secret", "exposed", "breaking", "alert",
    "hidden", "banned", "baffled", "unbelievable", "incredible", "amazing",
    "outrageous", "scandal", "conspiracy", "urgent", "exclusive", "bombshell",
    "leaked", "suppressed", "hoax", "plandemic", "illuminati",
    "clickbait", "viral", "censored", "silenced", "rigged", "stolen",
    "poisoned", "contaminated", "radiation",
    "big pharma", "they don't want", "share before", "wake up", "cover up",
    "mainstream media", "whistleblower", "deep state", "false flag",
    "one weird trick", "doctors hate", "won't believe", "must see",
    "they hide", "new world order", "microchip", "mind control",
    "population control", "government lies",
    "treason", "traitor", "destroy america", "sharia",
    "globalist", "soros", "libtard", "snowflake",
    "triggered", "owned", "destroyed", "slammed", "obliterated",
    "pathetic", "disgusting", "corrupt", "liar", "criminal",
    "truth about", "real reason", "they won't tell you",
    "share this", "spread the word", "wake up america",
    "open your eyes", "do your research", "brainwashed",
    "sheeple", "puppet", "agenda", "globalism", "cabal", "satanic",
]
HEDGE_WORDS = [
    "according to", "research shows", "study finds", "officials said",
    "reported", "stated", "confirmed", "announced", "published",
    "per cent", "percent", "data shows", "analysis", "evidence",
    "sources say", "spokesperson", "press release", "cited", "survey",
    "poll shows", "statistics", "researchers found", "scientists say",
    "experts warn", "government announced", "report says", "studies show",
    "trial results", "peer reviewed", "journal", "university", "institute",
]
FORMAL_WORDS = [
    "legislation", "parliamentary", "congressional", "administration",
    "spokesperson", "committee", "amendment", "regulatory", "fiscal",
    "monetary", "bilateral", "multilateral", "treaty", "ratified",
    "constitutional", "judiciary", "appellate", "plaintiff", "defendant",
    "indictment", "acquitted", "prosecuted", "arbitration",
    "sanction", "embargo", "tariff", "dividend", "quarterly", "annually",
]
EMOTIONAL_WORDS = [
    "terrifying", "disgusting", "horrifying", "evil", "destroy",
    "devastate", "catastrophic", "nightmare", "disaster", "chaos",
    "collapse", "invasion", "attack", "assault", "betrayed", "enslaved",
    "poisoning", "murdered", "executed", "genocide", "obliterate",
]

FEATURE_WEIGHTS = [
    4.5, 2.5, 2.5, 1.5, 2.0, 1.0, 1.5, 4.0,
    1.5, 2.0, 3.0, 3.0, 1.5, 1.5, 1.0, 1.0, 0.5, 2.0,
]
K_NEIGHBORS = 7


def extract_features(text):
    if not text or not text.strip():
        return [0.0] * 18
    lower   = text.lower()
    words   = text.split()
    n_words = max(len(words), 1)
    alpha   = [w.strip(string.punctuation) for w in words
               if w.strip(string.punctuation).isalpha()]
    n_alpha = max(len(alpha), 1)
    sents   = [s.strip() for s in
               text.replace('!', '.').replace('?', '.').split('.')
               if s.strip()]
    n_sents = max(len(sents), 1)

    f0  = sum(1 for w in SENSATIONAL_WORDS if w in lower) / n_words * 100
    f1  = sum(1 for w in words if len(w) > 2 and w.isupper()) / n_words
    f2  = text.count('!') / n_sents
    f3  = text.count('?') / n_sents
    f4  = sum(len(w) for w in alpha) / n_alpha
    f5  = math.log1p(n_sents)
    f6  = text.count('...')
    f7  = sum(1 for w in HEDGE_WORDS if w in lower) / n_sents
    f8  = sum(1 for ch in text if ch in string.punctuation) / len(text)
    unique = len(set(w.lower().strip(string.punctuation) for w in words))
    f9  = unique / n_words
    f10 = sum(1 for w in EMOTIONAL_WORDS if w in lower) / n_words * 100
    f11 = sum(1 for w in FORMAL_WORDS if w in lower) / n_words * 100
    f12 = sum(1 for ch in text if ch.isdigit()) / len(text)
    f13 = sum(1 for w in words
              if len(w) > 2 and w[0].isupper() and not w.isupper()) / n_words
    f14 = n_words / n_sents
    f15 = text.count('"') // 2 + text.count("'") // 2
    f16 = sum(1 for w in words if 'http' in w.lower() or 'www.' in w.lower())
    f17 = len(re.findall(r'[!?]{2,}|\.{3,}', text))
    return [round(x, 6) for x in
            [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17]]


class OnlineNormalizer:
    def __init__(self, k):
        self.k = k; self.n = 0
        self.mean = [0.0]*k; self.M2 = [0.0]*k

    def update(self, features):
        self.n += 1
        for i, x in enumerate(features):
            delta = x - self.mean[i]
            self.mean[i] += delta / self.n
            self.M2[i]   += delta * (x - self.mean[i])

    def std(self):
        if self.n < 2: return [1.0]*self.k
        return [math.sqrt(m/(self.n-1)) or 1.0 for m in self.M2]

    def normalize(self, features):
        stds = self.std()
        return [(x - self.mean[i])/stds[i] for i,x in enumerate(features)]


def apply_weights(f):
    return [x*w for x,w in zip(f, FEATURE_WEIGHTS)]

def prepare(text, norm):
    return apply_weights(norm.normalize(extract_features(text)))


class KDNode:
    def __init__(self, point, label, text, left=None, right=None):
        self.point=point; self.label=label; self.text=text
        self.left=left;   self.right=right

class KDTree:
    def __init__(self, k):
        self.k=k; self.root=None; self.size=0

    def _build(self, pts, depth=0):
        if not pts: return None
        axis = depth % self.k
        pts.sort(key=lambda x: x[0][axis])
        mid  = len(pts)//2
        node = KDNode(pts[mid][0], pts[mid][1], pts[mid][2])
        node.left  = self._build(pts[:mid],   depth+1)
        node.right = self._build(pts[mid+1:], depth+1)
        return node

    def build(self, data):
        self.root = self._build(data); self.size = len(data)

    @staticmethod
    def _dist(a, b):
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

    def knn(self, query, k=5):
        best = []
        def search(node, depth):
            if node is None: return
            d = self._dist(query, node.point)
            if len(best)<k:
                best.append((d,node.label,node.text))
                best.sort(key=lambda x:x[0], reverse=True)
            elif d < best[0][0]:
                best[0]=(d,node.label,node.text)
                best.sort(key=lambda x:x[0], reverse=True)
            axis = depth % self.k
            diff = query[axis]-node.point[axis]
            near,far = (node.left,node.right) if diff<0 else (node.right,node.left)
            search(near, depth+1)
            if len(best)<k or abs(diff)<best[0][0]:
                search(far, depth+1)
        search(self.root, 0)
        return sorted(best, key=lambda x:x[0])


FAKE_PRIOR = 1.3  # bias correction: boosts fake vote weight

def classify(tree, query_vec):
    nbrs = tree.knn(query_vec, k=K_NEIGHBORS)
    eps  = 1e-6
    w_real = sum(1.0/(d+eps) for d,l,_ in nbrs if l==1)
    w_fake = sum(1.0/(d+eps) for d,l,_ in nbrs if l==0)
    w_fake *= FAKE_PRIOR  # correct Real bias
    pred   = 1 if w_real >= w_fake else 0
    total  = w_real + w_fake
    conf   = round(max(w_real, w_fake) / total * 100)
    return pred, conf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.system("clear" if os.name == "posix" else "cls")
    dsep()
    print(c("  FAKE NEWS DETECTOR v2.0 — TEST SUITE", BOLD))
    print(c("  Stratified 20% Holdout · 18 Features · Weighted KNN · Fake-Prior Correction", GRAY))
    dsep()

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_FILE):
        print(c(f"\n  ERROR: {DATASET_FILE} not found.", RED))
        print(c("  Run setup_and_train.py first.\n", YELLOW))
        sys.exit(1)

    with open(DATASET_FILE) as f:
        raw = json.load(f)

    dataset = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            dataset.append((str(item[0]), int(item[1])))
        elif isinstance(item, dict):
            text  = item.get("text") or item.get("article") or ""
            label = int(item.get("label", item.get("class", 0)))
            dataset.append((text, label))

    print(c(f"\n  Dataset loaded     : {len(dataset)} articles", GREEN))

    # ── Stratified 20% split ──────────────────────────────────────────────────
    random.seed(42)
    fakes = [(t,l) for t,l in dataset if l==0]
    reals = [(t,l) for t,l in dataset if l==1]
    random.shuffle(fakes); random.shuffle(reals)

    n_test_fake = max(1, round(len(fakes)*0.20))
    n_test_real = max(1, round(len(reals)*0.20))
    test_set    = fakes[:n_test_fake]  + reals[:n_test_real]
    train_set   = fakes[n_test_fake:]  + reals[n_test_real:]
    random.shuffle(test_set); random.shuffle(train_set)

    print(c(f"  Training articles  : {len(train_set)} "
            f"({len(train_set)-n_test_real*4} fake + {len(reals)-n_test_real} real)", CYAN))
    print(c(f"  Test articles      : {len(test_set)} "
            f"({n_test_fake} fake + {n_test_real} real)", CYAN))
    sep()

    # ── Fit normalizer on training split ─────────────────────────────────────
    print(c("\n  Fitting normalizer on training split...", BLUE))
    norm = OnlineNormalizer(k=18)
    for text, _ in train_set:
        norm.update(extract_features(text))
    print(c(f"  Normalizer fitted on {norm.n} samples.", GREEN))

    # ── Build KD-Tree ─────────────────────────────────────────────────────────
    print(c("  Building KD-Tree...", BLUE))
    t0 = time.perf_counter()
    tree_data = [(prepare(t, norm), l, t) for t,l in train_set]
    tree = KDTree(k=18)
    tree.build(tree_data)
    build_ms = (time.perf_counter()-t0)*1000
    print(c(f"  KD-Tree built in {build_ms:.1f} ms  "
            f"({tree.size} nodes, 18 dims, k={K_NEIGHBORS})", GREEN))
    sep()

    # ── Run predictions ───────────────────────────────────────────────────────
    print(c("\n  Running predictions...\n", BLUE))
    results   = []
    latencies = []

    for i, (text, true_label) in enumerate(test_set):
        t1 = time.perf_counter()
        qv = prepare(text, norm)
        pred, conf = classify(tree, qv)
        latencies.append((time.perf_counter()-t1)*1000)
        results.append((true_label, pred, conf, text))

        done = int((i+1)/len(test_set)*40)
        bar  = "█"*done + "░"*(40-done)
        running = sum(1 for tr,pr,_,_ in results if tr==pr)/(i+1)*100
        print(f"\r  [{bar}] {i+1}/{len(test_set)}  "
              f"acc: {c(f'{running:.1f}%', YELLOW)}", end="", flush=True)

    print()

    # ── Metrics ───────────────────────────────────────────────────────────────
    TP = sum(1 for tr,pr,_,_ in results if tr==1 and pr==1)
    TN = sum(1 for tr,pr,_,_ in results if tr==0 and pr==0)
    FP = sum(1 for tr,pr,_,_ in results if tr==0 and pr==1)
    FN = sum(1 for tr,pr,_,_ in results if tr==1 and pr==0)
    total   = len(results)
    correct = TP+TN
    accuracy = correct/total*100

    prec_fake = TN/(TN+FN) if (TN+FN)>0 else 0
    rec_fake  = TN/(TN+FP) if (TN+FP)>0 else 0
    f1_fake   = 2*prec_fake*rec_fake/(prec_fake+rec_fake) if (prec_fake+rec_fake)>0 else 0
    prec_real = TP/(TP+FP) if (TP+FP)>0 else 0
    rec_real  = TP/(TP+FN) if (TP+FN)>0 else 0
    f1_real   = 2*prec_real*rec_real/(prec_real+rec_real) if (prec_real+rec_real)>0 else 0
    macro_f1  = (f1_fake+f1_real)/2
    avg_lat   = sum(latencies)/len(latencies)
    avg_conf  = sum(cf for _,_,cf,_ in results)/total

    # ── Print report ──────────────────────────────────────────────────────────
    sep()
    print()
    dsep()
    print(c("  RESULTS", BOLD))
    dsep()
    print()
    acc_color = GREEN if accuracy>=80 else YELLOW if accuracy>=70 else RED
    print(c(f"  {'ACCURACY':<26} {accuracy:.2f}%", acc_color+BOLD))

    sep()

    print(c(f"  {'Metric':<28} {'Fake (0)':<16} {'Real (1)'}", BOLD))
    sep()
    print(f"  {'Precision':<28} "
          f"{c(f'{prec_fake:.4f}', PURPLE):<23} "
          f"{c(f'{prec_real:.4f}', PURPLE)}")
    print(f"  {'Recall':<28} "
          f"{c(f'{rec_fake:.4f}', CYAN):<23} "
          f"{c(f'{rec_real:.4f}', CYAN)}")
    print(f"  {'F1 Score':<28} "
          f"{c(f'{f1_fake:.4f}', YELLOW):<23} "
          f"{c(f'{f1_real:.4f}', YELLOW)}")
    sep()
    print(f"  {'Macro F1':<28} {c(f'{macro_f1:.4f}', GREEN+BOLD)}")
    print()
    sep()

    # Confusion matrix
    print(c("  Confusion Matrix (Predicted →)", BOLD))
    print()
    print(c("                        Fake      Real", GRAY))
    print(f"  {c('Actual', GRAY)}  {c('Fake', RED)}    "
          f"[ {c(str(TN).rjust(4), GREEN)} ]  [ {c(str(FP).rjust(4), RED)} ]   "
          f"{c(f'({rec_fake*100:.1f}% recall)', GRAY)}")
    print(f"          {c('Real', GREEN)}    "
          f"[ {c(str(FN).rjust(4), RED)} ]  [ {c(str(TP).rjust(4), GREEN)} ]   "
          f"{c(f'({rec_real*100:.1f}% recall)', GRAY)}")
    print()
    sep()

    # Performance
    print(c("  Performance", BOLD))
    print(f"  {'Avg query latency':<28} {c(f'{avg_lat:.3f} ms', CYAN)}")
    print(f"  {'Tree build time':<28} {c(f'{build_ms:.1f} ms', CYAN)}")
    print(f"  {'Avg confidence score':<28} {c(f'{avg_conf:.1f}%', YELLOW)}")
    print(f"  {'Total test predictions':<28} {c(str(total), BLUE)}")
    print(f"  {'Feature dimensions':<28} {c('18', BLUE)}")
    print(f"  {'k (neighbors)':<28} {c(str(K_NEIGHBORS), BLUE)}")
    sep()

    # Top 5 mistakes
    mistakes = [(tr,pr,cf,tx) for tr,pr,cf,tx in results if tr!=pr]
    mistakes.sort(key=lambda x: x[2])

    if mistakes:
        print()
        print(c(f"  MOST CONFUSED PREDICTIONS  ({len(mistakes)} total errors)", BOLD))
        sep()
        for tr,pr,cf,tx in mistakes[:5]:
            true_s = c("FAKE", RED)  if tr==0 else c("REAL", GREEN)
            pred_s = c("FAKE", RED)  if pr==0 else c("REAL", GREEN)
            snippet = (tx[:70]+"...") if len(tx)>70 else tx
            print(f"  True: {true_s}  Predicted: {pred_s}  conf {c(str(cf)+'%', YELLOW)}")
            print(c(f"  \"{snippet}\"", GRAY))
            print()
    else:
        print()
        print(c("  No errors — perfect score on this test set!", GREEN+BOLD))

    sep()

    # Confidence distribution
    bins = {}
    for _,_,cf,_ in results:
        bins[cf] = bins.get(cf,0)+1
    print()
    print(c("  Confidence Distribution", BOLD))
    sep()
    for cf_val in sorted(bins.keys(), reverse=True):
        cnt = bins[cf_val]
        pct = cnt/total*100
        bar = "█" * int(pct/2)
        print(f"  {str(cf_val)+'%':<6}  {bar:<50} {cnt:>4} ({pct:.1f}%)")
    sep()
    print()
    dsep()
    print(c("  TEST COMPLETE", BOLD + (GREEN if accuracy>=80 else YELLOW)))
    dsep()
    print()


if __name__ == "__main__":
    main()
