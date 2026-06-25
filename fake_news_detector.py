
import math
import string
import json
import os
import re
from datetime import datetime


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
    # Political/tabloid language dominant in McIntire & ISOT fake corpora
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

FEATURE_NAMES = [
    "sensational_density",
    "caps_ratio",
    "exclamation_density",
    "question_density",
    "avg_word_length",
    "sentence_count_log",
    "ellipsis_count",
    "hedge_density",
    "punct_density",
    "lexical_diversity",
    "emotional_density",
    "formal_word_density",
    "digit_ratio",
    "title_case_ratio",
    "avg_sentence_length",
    "quote_count",
    "url_count",
    "repeated_punct",
]

FEATURE_WEIGHTS = [
    4.5,   # 0  sensational_density      ← boosted: strongest fake signal
    2.5,   # 1  caps_ratio
    2.5,   # 2  exclamation_density
    1.5,   # 3  question_density
    2.0,   # 4  avg_word_length
    1.0,   # 5  sentence_count_log
    1.5,   # 6  ellipsis_count
    4.0,   # 7  hedge_density            ← boosted: strongest real signal
    1.5,   # 8  punct_density
    2.0,   # 9  lexical_diversity
    3.0,   # 10 emotional_density        ← boosted
    3.0,   # 11 formal_word_density      ← boosted
    1.5,   # 12 digit_ratio
    1.5,   # 13 title_case_ratio
    1.0,   # 14 avg_sentence_length
    1.0,   # 15 quote_count
    0.5,   # 16 url_count
    2.0,   # 17 repeated_punct
]

K_NEIGHBORS = 7


#FEATURE EXTRACTION

def extract_features(text: str) -> list:
    if not text or not text.strip():
        return [0.0] * 18

    lower   = text.lower()
    words   = text.split()
    n_words = max(len(words), 1)

    alpha = [w.strip(string.punctuation) for w in words
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
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
             f10, f11, f12, f13, f14, f15, f16, f17]]


# ONLINE NORMALIZER  (Welford's algorithm)

class OnlineNormalizer:
    def __init__(self, k: int):
        self.k    = k
        self.n    = 0
        self.mean = [0.0] * k
        self.M2   = [0.0] * k

    def update(self, features: list):
        self.n += 1
        for i, x in enumerate(features):
            delta       = x - self.mean[i]
            self.mean[i] += delta / self.n
            self.M2[i]  += delta * (x - self.mean[i])

    def std(self) -> list:
        if self.n < 2:
            return [1.0] * self.k
        return [math.sqrt(m / (self.n - 1)) or 1.0 for m in self.M2]

    def normalize(self, features: list) -> list:
        stds = self.std()
        return [(x - self.mean[i]) / stds[i] for i, x in enumerate(features)]

    def to_dict(self):
        return {"n": self.n, "mean": self.mean, "M2": self.M2, "k": self.k}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["k"])
        obj.n    = d["n"]
        obj.mean = d["mean"]
        obj.M2   = d["M2"]
        return obj


# KD-TREE

class KDNode:
    def __init__(self, point, label, text, left=None, right=None):
        self.point = point
        self.label = label
        self.text  = text
        self.left  = left
        self.right = right


class KDTree:
    def __init__(self, k: int):
        self.k    = k
        self.root = None
        self.size = 0

    def _build(self, points, depth=0):
        if not points:
            return None
        axis = depth % self.k
        points.sort(key=lambda x: x[0][axis])
        mid  = len(points) // 2
        node = KDNode(points[mid][0], points[mid][1], points[mid][2])
        node.left  = self._build(points[:mid],   depth + 1)
        node.right = self._build(points[mid+1:], depth + 1)
        return node

    def build(self, data):
        self.root = self._build(data)
        self.size = len(data)

    def insert(self, point, label, text):
        self.root  = self._ins(self.root, point, label, text, 0)
        self.size += 1

    def _ins(self, node, point, label, text, depth):
        if node is None:
            return KDNode(point, label, text)
        axis = depth % self.k
        if point[axis] < node.point[axis]:
            node.left  = self._ins(node.left,  point, label, text, depth + 1)
        else:
            node.right = self._ins(node.right, point, label, text, depth + 1)
        return node

    @staticmethod
    def _euclidean(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def knn_search(self, query, k=5):
        best = []

        def search(node, depth):
            if node is None:
                return
            dist = self._euclidean(query, node.point)
            if len(best) < k:
                best.append((dist, node.label, node.text))
                best.sort(key=lambda x: x[0], reverse=True)
            elif dist < best[0][0]:
                best[0] = (dist, node.label, node.text)
                best.sort(key=lambda x: x[0], reverse=True)
            axis = depth % self.k
            diff = query[axis] - node.point[axis]
            close, away = ((node.left, node.right) if diff < 0
                           else (node.right, node.left))
            search(close, depth + 1)
            if len(best) < k or abs(diff) < best[0][0]:
                search(away, depth + 1)

        search(self.root, 0)
        return sorted(best, key=lambda x: x[0])


# SEED DATASET

SEED_ARTICLES = [
    ("SHOCKING: Scientists BAFFLED as miracle cure for ALL cancers found in common household item!!! Big Pharma HIDING this!!!", 0),
    ("BREAKING ALERT!!! Government secretly putting mind control chips in vaccines!! Whistleblower EXPOSES everything!!!! Share before DELETE!!!", 0),
    ("You WON'T BELIEVE what this celebrity did!!! Doctors are OUTRAGED and the TRUTH will SHOCK you!!! MUST SEE!!!!", 0),
    ("LEAKED document EXPOSES global conspiracy!!! They don't want you to know THIS secret!!! Wake up sheeple!!!", 0),
    ("ONE WEIRD TRICK banned by doctors!!! Mainstream media covering up the AMAZING truth about your health!!!", 0),
    ("The Federal Reserve raised interest rates by 25 basis points on Wednesday, citing continued concerns about inflation while acknowledging slowing economic growth.", 1),
    ("Researchers at Stanford University published a study in Nature Medicine showing a potential link between sleep deprivation and increased risk of cardiovascular disease.", 1),
    ("The city council voted 7-2 to approve the new public transit expansion plan, which includes three new bus routes and extended evening service.", 1),
    ("According to data released by the Census Bureau, the national unemployment rate fell to 3.7 percent last month, down from 3.9 percent in the previous quarter.", 1),
    ("Scientists confirmed the discovery of a new exoplanet approximately 1.4 times the size of Earth orbiting within the habitable zone of its star, according to findings published in The Astrophysical Journal.", 1),
]


# 6. PREPARE VECTOR  (normalize + weight)

def apply_weights(features: list) -> list:
    return [f * w for f, w in zip(features, FEATURE_WEIGHTS)]


def prepare(text: str, normalizer: OnlineNormalizer) -> list:
    raw     = extract_features(text)
    normed  = normalizer.normalize(raw)
    return apply_weights(normed)


# PREDICTION ENGINE  (distance-weighted voting)

FAKE_PRIOR = 1.3

def predict(tree: KDTree, query_vec: list) -> dict:
    neighbors    = tree.knn_search(query_vec, k=K_NEIGHBORS)
    eps          = 1e-6
    weight_real  = sum(1.0 / (d + eps) for d, l, _ in neighbors if l == 1)
    weight_fake  = sum(1.0 / (d + eps) for d, l, _ in neighbors if l == 0)
    # Apply prior: amplify fake signal to correct systematic Real bias
    weight_fake *= FAKE_PRIOR
    predicted    = 1 if weight_real >= weight_fake else 0
    total_w      = weight_real + weight_fake
    confidence   = round(max(weight_real, weight_fake) / total_w * 100)
    return {
        "label":       predicted,
        "label_text":  "REAL" if predicted == 1 else "FAKE",
        "confidence":  confidence,
        "weight_real": round(weight_real, 4),
        "weight_fake": round(weight_fake, 4),
        "votes_real":  sum(1 for _, l, _ in neighbors if l == 1),
        "votes_fake":  sum(1 for _, l, _ in neighbors if l == 0),
        "neighbors":   neighbors,
    }


# EXPLAINABILITY

def explain(raw_features: list, prediction: dict) -> list:
    reasons = []
    f = raw_features

    if f[0] >= 2.0:
        reasons.append(f"Very high sensational keyword density ({f[0]:.1f} per 100 words)")
    elif f[0] >= 0.5:
        reasons.append(f"Contains sensational/clickbait keywords (density {f[0]:.1f})")
    if f[1] >= 0.05:
        reasons.append(f"Elevated ALL-CAPS ratio ({f[1]*100:.1f}% of words)")
    if f[2] >= 1.5:
        reasons.append(f"High exclamation density ({f[2]:.1f} per sentence)")
    if f[17] >= 2:
        reasons.append(f"Contains {int(f[17])} repeated punctuation bursts (!!!/???)")
    if f[4] < 4.3 and prediction["label"] == 0:
        reasons.append(f"Short avg word length ({f[4]:.1f} chars) — simplified vocabulary")
    elif f[4] > 5.8 and prediction["label"] == 1:
        reasons.append(f"Long avg word length ({f[4]:.1f} chars) — formal vocabulary")
    if f[7] == 0 and prediction["label"] == 0:
        reasons.append("No credibility signals or source attributions detected")
    elif f[7] >= 0.3:
        reasons.append(f"Strong credibility signals present (density {f[7]:.2f})")
    if f[11] >= 0.5:
        reasons.append(f"Contains formal journalism terminology (density {f[11]:.1f})")
    if f[10] >= 0.8:
        reasons.append(f"High emotional language density ({f[10]:.1f} per 100 words)")
    if f[9] < 0.55 and prediction["label"] == 0:
        reasons.append(f"Low lexical diversity ({f[9]:.2f}) — repetitive language")
    elif f[9] > 0.75 and prediction["label"] == 1:
        reasons.append(f"High lexical diversity ({f[9]:.2f}) — varied vocabulary")
    fk = prediction["votes_fake"]
    rl = prediction["votes_real"]
    if fk >= 3:
        reasons.append(f"Strongly matched {fk}/{K_NEIGHBORS} fake articles in KD-Tree")
    elif rl >= 3:
        reasons.append(f"Strongly matched {rl}/{K_NEIGHBORS} real articles in KD-Tree")
    elif fk > rl:
        reasons.append(f"Majority vote: {fk} fake vs {rl} real neighbors")
    elif rl > fk:
        reasons.append(f"Majority vote: {rl} real vs {fk} fake neighbors")
    if not reasons:
        reasons.append(f"Feature pattern closest to {prediction['label_text'].lower()} "
                       f"cluster (confidence {prediction['confidence']}%)")
    return reasons


# 9. PERSISTENCE

DATA_FILE = "fake_news_dataset.json"
NORM_FILE = "fake_news_normalizer.json"


def save_dataset(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save_normalizer(norm):
    with open(NORM_FILE, "w") as f:
        json.dump(norm.to_dict(), f, indent=2)


def load_normalizer(k):
    if os.path.exists(NORM_FILE):
        with open(NORM_FILE) as f:
            return OnlineNormalizer.from_dict(json.load(f))
    return OnlineNormalizer(k)


#  DISPLAY

RESET  = "\033[0m"; BOLD   = "\033[1m"
RED    = "\033[91m"; GREEN  = "\033[92m"
YELLOW = "\033[93m"; BLUE   = "\033[94m"
PURPLE = "\033[95m"; CYAN   = "\033[96m"
GRAY   = "\033[90m"


def c(text, color): return f"{color}{text}{RESET}"
def sep():  print(c("─" * 62, GRAY))
def dsep(): print(c("═" * 62, CYAN))


def banner():
    dsep()
    print(c("  Fake News Detection System v2.0", BOLD))
    print(c("  KD-Tree + Weighted KNN + Incremental Learning", GRAY))
    dsep()


def print_features(raw):
    print(c("\n  Feature Vector (raw):", BLUE))
    for name, val in zip(FEATURE_NAMES, raw):
        bar = "█" * min(int(val * 6), 24)
        print(f"    {name:<26} {c(f'{val:.4f}'.ljust(8), PURPLE)}  {c(bar, CYAN)}")


def print_neighbors(neighbors):
    print(c(f"\n  KD-Tree Nearest Neighbors (k={K_NEIGHBORS}):", BLUE))
    for i, (dist, label, text) in enumerate(neighbors):
        lstr    = c("[FAKE]", RED) if label == 0 else c("[REAL]", GREEN)
        snippet = (text[:52] + "...") if len(text) > 52 else text
        print(f"    {i+1}. {lstr}  dist={dist:.4f}  \"{c(snippet, GRAY)}\"")


def print_prediction(prediction):
    color = RED if prediction["label"] == 0 else GREEN
    print()
    sep()
    print(c(f"\n  PREDICTION:  [ {prediction['label_text']} ]", color + BOLD))
    print(f"  Confidence : {c(str(prediction['confidence']) + '%', YELLOW)}")
    print(f"  Weighted   → "
          f"{c('Fake: ' + str(prediction['weight_fake']), RED)}  "
          f"{c('Real: ' + str(prediction['weight_real']), GREEN)}")
    print(f"  Raw votes  → "
          f"{c('Fake: ' + str(prediction['votes_fake']), RED)}  "
          f"{c('Real: ' + str(prediction['votes_real']), GREEN)}")


def print_explanation(reasons):
    print(c("\n  Reasons:", BLUE))
    for r in reasons:
        print(f"    • {c(r, YELLOW)}")



# MAIN LOOP

def main():
    os.system("clear" if os.name == "posix" else "cls")
    banner()

    k_dims     = len(FEATURE_NAMES)
    normalizer = load_normalizer(k_dims)

    saved = load_dataset()
    if saved:
        raw_data = [(str(item[0]), int(item[1]))
                    for item in saved
                    if isinstance(item, (list, tuple)) and len(item) == 2]
        print(c(f"\n  Loaded {len(raw_data)} articles from {DATA_FILE}", GREEN))
    else:
        raw_data = list(SEED_ARTICLES)
        print(c(f"\n  Initialized with {len(raw_data)} seed articles.", CYAN))

    if normalizer.n == 0:
        print(c("  Fitting normalizer on dataset...", GRAY))
        for text, _ in raw_data:
            normalizer.update(extract_features(text))
        save_normalizer(normalizer)

    print(c("  Building KD-Tree...", GRAY))
    tree      = KDTree(k=k_dims)
    tree_data = [(prepare(text, normalizer), label, text)
                 for text, label in raw_data]
    tree.build(tree_data)
    dataset = list(raw_data)

    stats = {"analyzed": 0, "fake": 0, "real": 0, "learned": 0}
    print(c(f"  KD-Tree ready: {tree.size} nodes, {k_dims} dims, k={K_NEIGHBORS}.", CYAN))
    sep()

    while True:
        print(c("\n  [ OPTIONS ]", BOLD))
        print("    1. Analyze an article")
        print("    2. Show dataset statistics")
        print("    3. Export dataset")
        print("    4. Quit")
        print()

        choice = input(c("  > ", GREEN)).strip()

        if choice == "4" or choice.lower() in ("q", "quit", "exit"):
            print(c("\n  Goodbye! Saving state...\n", CYAN))
            save_dataset(dataset)
            save_normalizer(normalizer)
            break

        elif choice == "2":
            sep()
            print(c("\n  Dataset Statistics:", BLUE))
            print(f"    Total articles    : {len(dataset)}")
            print(f"    Fake articles     : {sum(1 for _, l in dataset if l == 0)}")
            print(f"    Real articles     : {sum(1 for _, l in dataset if l == 1)}")
            print(f"    Analyzed (session): {stats['analyzed']}")
            print(f"    Learned (session) : {stats['learned']}")
            print(f"    Normalizer samples: {normalizer.n}")
            sep()

        elif choice == "3":
            save_dataset(dataset)
            save_normalizer(normalizer)
            print(c(f"\n  Exported: {DATA_FILE} + {NORM_FILE}", GREEN))

        elif choice == "1":
            print()
            sep()
            print(c("  Paste your news article (press Enter twice when done):", BLUE))
            print(c("  > ", GREEN), end="", flush=True)

            lines = []
            try:
                while True:
                    line = input()
                    if line == "" and lines:
                        break
                    lines.append(line)
            except EOFError:
                pass

            article = " ".join(lines).strip()
            if not article:
                print(c("\n  No text entered. Try again.", RED))
                continue

            print(c("\n  Extracting features...", GRAY))
            raw_feats = extract_features(article)
            query_vec = prepare(article, normalizer)

            print_features(raw_feats)
            print(c("\n  Searching KD-Tree...", GRAY))
            result  = predict(tree, query_vec)
            reasons = explain(raw_feats, result)
            print_neighbors(result["neighbors"])
            print_prediction(result)
            print_explanation(reasons)
            sep()

            stats["analyzed"] += 1
            stats["fake" if result["label"] == 0 else "real"] += 1

            print()
            feedback = input(c("  Was this prediction correct? (y/n): ", YELLOW)).strip().lower()

            if feedback == "y":
                correct_label = result["label"]
                print(c("\n  Feedback recorded.", GREEN))
            elif feedback == "n":
                while True:
                    flip = input(c("  Enter correct label [0=Fake / 1=Real]: ", YELLOW)).strip()
                    if flip in ("0", "1"):
                        correct_label = int(flip)
                        break
                    print(c("  Please enter 0 or 1.", RED))
                print(c(f"\n  Corrected to: {'REAL' if correct_label == 1 else 'FAKE'}", GREEN))
            else:
                print(c("  Skipping feedback.", GRAY))
                continue

            normalizer.update(raw_feats)
            new_vec = prepare(article, normalizer)
            tree.insert(new_vec, correct_label, article)
            dataset.append((article, correct_label))
            save_dataset(dataset)
            save_normalizer(normalizer)
            stats["learned"] += 1

            print(c(f"  KD-Tree updated — {tree.size} articles.", GREEN))
            print(c(f"  Normalizer updated — {normalizer.n} samples.", GREEN))
            print(c(f"  Saved at {datetime.now().strftime('%H:%M:%S')}", GRAY))
            sep()
        else:
            print(c("  Invalid option. Enter 1, 2, 3, or 4.", RED))


if __name__ == "__main__":
    main()
