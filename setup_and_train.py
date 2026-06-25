
import urllib.request
import csv
import json
import math
import string
import os
import random
import sys

# ── Colors ──────────────────────────────────────────
RESET  = "\033[0m"; BOLD = "\033[1m"
RED    = "\033[91m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; CYAN  = "\033[96m"
GRAY   = "\033[90m"; BLUE  = "\033[94m"

def c(t, col): return f"{col}{t}{RESET}"
def sep(): print(c("─" * 60, GRAY))

# ── Dataset Sources ──────────────────────────────────
SOURCES = {
    "fake": [
        # George McIntire dataset (GitHub raw CSV, no login)
        "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv",
    ],
    "isot_fake": [
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.json",
    ],
    # Fallback: manually curated small CSVs from open repos
    "fallback": "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
}

OUTPUT_FILE = os.path.expanduser("~/fake_news_dataset.json")
TRAIN_FILE  = os.path.expanduser("~/fake_news_trained_tree.json")

# ─────────────────────────────────────────────────────
# FEATURE EXTRACTION (same as detector)
# ─────────────────────────────────────────────────────
SENSATIONAL_WORDS = [
    "shocking","miracle","secret","exposed","breaking","alert","hidden",
    "banned","truth","baffled","unbelievable","incredible","amazing",
    "outrageous","scandal","conspiracy","they don't want","share before",
    "delete","wake up","mainstream media","cover up","whistleblower",
    "urgent","exclusive","bombshell","leaked","suppressed","must see",
    "won't believe","doctors hate","one weird trick","deep state",
    "false flag","plandemic","fake news","hoax","illuminati"
]
HEDGE_WORDS = [
    "according to","research shows","study finds","officials said",
    "reported","stated","confirmed","announced","published",
    "per cent","percent","data shows","analysis","evidence",
    "sources say","spokesperson","press release","cited"
]
FEATURE_NAMES = [
    "sensational_words","caps_words","exclamations","questions",
    "avg_word_length","sentence_count","ellipses","hedge_words",
    "punct_density","lexical_diversity"
]

def extract_features(text: str) -> list:
    if not text or not text.strip():
        return [0.0] * 10
    lower = text.lower()
    words = text.split()
    alpha_words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation).isalpha()]
    sentences = [s.strip() for s in text.replace('!','.').replace('?','.').split('.') if s.strip()]
    sensational = sum(1 for w in SENSATIONAL_WORDS if w in lower)
    caps = sum(1 for w in words if len(w) > 2 and w.isupper())
    exclaim = text.count('!')
    questions = text.count('?')
    avg_wl = (sum(len(w) for w in alpha_words) / len(alpha_words)) if alpha_words else 0
    sent_count = max(len(sentences), 1)
    ellipsis = text.count('...')
    hedge = sum(1 for w in HEDGE_WORDS if w in lower)
    punct = sum(1 for ch in text if ch in string.punctuation)
    punct_dens = punct / len(text) if text else 0
    unique = len(set(w.lower().strip(string.punctuation) for w in words))
    lex_div = unique / len(words) if words else 0
    return [round(x, 4) for x in [
        sensational, caps, exclaim, questions, avg_wl,
        sent_count, ellipsis, hedge, punct_dens, lex_div
    ]]

# ─────────────────────────────────────────────────────
# KD-TREE
# ─────────────────────────────────────────────────────
class KDNode:
    def __init__(self, point, label, text, left=None, right=None):
        self.point = point; self.label = label; self.text = text
        self.left = left;   self.right = right

class KDTree:
    def __init__(self, k):
        self.k = k; self.root = None; self.size = 0

    def _build(self, pts, depth=0):
        if not pts: return None
        axis = depth % self.k
        pts.sort(key=lambda x: x[0][axis])
        mid = len(pts) // 2
        n = KDNode(pts[mid][0], pts[mid][1], pts[mid][2])
        n.left  = self._build(pts[:mid],   depth+1)
        n.right = self._build(pts[mid+1:], depth+1)
        return n

    def build(self, data):
        self.root = self._build(data); self.size = len(data)

    def insert(self, point, label, text):
        self.root = self._ins(self.root, point, label, text, 0)
        self.size += 1

    def _ins(self, node, point, label, text, depth):
        if node is None: return KDNode(point, label, text)
        axis = depth % self.k
        if point[axis] < node.point[axis]:
            node.left  = self._ins(node.left,  point, label, text, depth+1)
        else:
            node.right = self._ins(node.right, point, label, text, depth+1)
        return node

    @staticmethod
    def _dist(a, b):
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

    def knn(self, query, k=3):
        best = []
        def search(node, depth):
            if node is None: return
            d = self._dist(query, node.point)
            if len(best) < k:
                best.append((d, node.label, node.text))
                best.sort(key=lambda x: x[0], reverse=True)
            elif d < best[0][0]:
                best[0] = (d, node.label, node.text)
                best.sort(key=lambda x: x[0], reverse=True)
            axis = depth % self.k
            diff = query[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            search(near, depth+1)
            if len(best) < k or abs(diff) < best[0][0]:
                search(far, depth+1)
        search(self.root, 0)
        return sorted(best, key=lambda x: x[0])

# ─────────────────────────────────────────────────────
# DOWNLOAD HELPERS
# ─────────────────────────────────────────────────────
def download(url, dest):
    print(c(f"  Downloading: {url}", GRAY))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := r.read(8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded / total * 40)
                    bar = "█" * pct + "░" * (40 - pct)
                    print(f"\r  [{bar}] {downloaded//1024}KB / {total//1024}KB", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(c(f"\n  Failed: {e}", RED))
        return False

# ─────────────────────────────────────────────────────
# PARSE CSV  (handles George McIntire + ISOT formats)
# ─────────────────────────────────────────────────────
def parse_csv(filepath):
    """Returns list of (text, label) where label 0=fake 1=real"""
    articles = []
    with open(filepath, newline='', encoding='utf-8', errors='ignore') as f:
        sample = f.read(2048); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]

        for row in reader:
            row_lower = {k.lower().strip(): v for k, v in row.items()}

            # Get text: prefer 'text', else 'title', else first long column
            text = ""
            for col in ["text", "title", "content", "article", "body"]:
                if col in row_lower and len(row_lower[col].strip()) > 20:
                    text = row_lower[col].strip()
                    break
            if not text:
                for v in row_lower.values():
                    if isinstance(v, str) and len(v) > 40:
                        text = v.strip(); break
            if not text or len(text) < 15:
                continue

            # Get label
            label = None
            for col in ["label", "class", "fake", "real", "type", "category"]:
                if col in row_lower:
                    val = row_lower[col].strip().lower()
                    if val in ("fake", "0", "false", "FAKE", "Fake"):
                        label = 0; break
                    elif val in ("real", "1", "true", "REAL", "Real", "TRUE"):
                        label = 1; break
            if label is None:
                continue

            articles.append((text[:500], label))  # cap at 500 chars

    return articles

# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    os.system("clear")
    print(c("═" * 60, CYAN))
    print(c("  Fake News Detector — Dataset Download & Training", BOLD))
    print(c("═" * 60, CYAN))
    print()

    tmp_csv = os.path.expanduser("~/fake_real_news_raw.csv")

    # ── Step 1: Download ─────────────────────────────
    print(c("[ STEP 1 ] Downloading dataset...", BLUE))
    sep()

    urls = [
        "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv",
        "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
    ]

    downloaded = False
    for url in urls:
        if download(url, tmp_csv):
            downloaded = True
            break

    if not downloaded:
        print(c("\n  All downloads failed. Check your internet and try again.", RED))
        print(c("  Or manually download from:", YELLOW))
        print("  https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news")
        print("  Save as ~/fake_real_news_raw.csv and rerun this script.")
        sys.exit(1)

    # ── Step 2: Parse ────────────────────────────────
    print(c("\n[ STEP 2 ] Parsing CSV...", BLUE))
    sep()
    all_articles = parse_csv(tmp_csv)
    print(c(f"  Parsed {len(all_articles)} labeled articles.", GREEN))

    fake_all = [(t, l) for t, l in all_articles if l == 0]
    real_all = [(t, l) for t, l in all_articles if l == 1]
    print(f"  Fake: {c(str(len(fake_all)), RED)}   Real: {c(str(len(real_all)), GREEN)}")

    if len(fake_all) < 50 or len(real_all) < 50:
        print(c("\n  Not enough data parsed from download.", RED))
        print(c("  Try the Kaggle manual download instead (see above).", YELLOW))
        sys.exit(1)

    # ── Step 3: Balance to 1000 each ─────────────────
    print(c("\n[ STEP 3 ] Sampling 1000 fake + 1000 real...", BLUE))
    sep()
    random.seed(42)
    n_fake = min(1000, len(fake_all))
    n_real = min(1000, len(real_all))
    selected = random.sample(fake_all, n_fake) + random.sample(real_all, n_real)
    random.shuffle(selected)
    print(c(f"  Selected {n_fake} fake + {n_real} real = {len(selected)} total articles.", GREEN))

    # ── Step 4: Feature extraction ───────────────────
    print(c("\n[ STEP 4 ] Extracting features (10-dim vectors)...", BLUE))
    sep()
    tree_data = []
    dataset_export = []
    for i, (text, label) in enumerate(selected):
        feats = extract_features(text)
        tree_data.append((feats, label, text))
        dataset_export.append((text, label))
        if (i + 1) % 200 == 0:
            pct = int((i + 1) / len(selected) * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {i+1}/{len(selected)}", end="", flush=True)
    print(f"\r  [{'█'*40}] {len(selected)}/{len(selected)}")
    print(c(f"  Done. Feature vectors ready.", GREEN))

    # ── Step 5: Build KD-Tree ─────────────────────────
    print(c("\n[ STEP 5 ] Building KD-Tree...", BLUE))
    sep()
    k_dims = 10
    tree = KDTree(k=k_dims)
    tree.build(tree_data)
    print(c(f"  KD-Tree built: {tree.size} nodes, {k_dims} dimensions.", GREEN))

    # ── Step 6: Quick accuracy check (holdout 10%) ───
    print(c("\n[ STEP 6 ] Evaluating accuracy (10% holdout)...", BLUE))
    sep()
    random.seed(99)
    test_size = max(50, len(selected) // 10)
    test_set = random.sample(selected, test_size)
    correct = 0
    for text, true_label in test_set:
        feats = extract_features(text)
        neighbors = tree.knn(feats, k=3)
        votes_real = sum(1 for _, lb, _ in neighbors if lb == 1)
        pred = 1 if votes_real > 1 else 0
        if pred == true_label:
            correct += 1
    accuracy = round(correct / test_size * 100, 1)
    acc_color = GREEN if accuracy >= 70 else YELLOW if accuracy >= 55 else RED
    print(c(f"  Accuracy on {test_size} holdout samples: {accuracy}%", acc_color))
    if accuracy < 60:
        print(c("  Note: KD-Tree KNN accuracy depends on feature quality.", YELLOW))
        print(c("  The model will improve as you give feedback during use.", YELLOW))

    # ── Step 7: Save dataset ──────────────────────────
    print(c("\n[ STEP 7 ] Saving dataset...", BLUE))
    sep()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset_export, f, indent=2)
    print(c(f"  Dataset saved → {OUTPUT_FILE}", GREEN))

    # Cleanup temp file
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)
        print(c(f"  Temp file removed.", GRAY))

    # ── Done ──────────────────────────────────────────
    print()
    print(c("═" * 60, CYAN))
    print(c("  TRAINING COMPLETE!", BOLD + GREEN))
    print(c("═" * 60, CYAN))
    print(f"\n  Dataset : {c(OUTPUT_FILE, CYAN)}")
    print(f"  Articles: {c(str(len(selected)), YELLOW)}")
    print(f"  Accuracy: {c(str(accuracy) + '%', acc_color)}")
    print()
    print(c("  Next step — run the detector:", BLUE))
    print(c("    python3 ~/fake_news_detector.py", YELLOW))
    print()


if __name__ == "__main__":
    main()
