"""Dual-branch KC discovery for programming knowledge tracing.

KC identity is encoded from code structure ONLY. Score/reason/compile labels
supervise training and describe evidence, never scale clustering coordinates.
Fit on training students, freeze artifacts, then attribute individual submissions.

Examples (run from model/MLKT4 after installing the replacement):
  python ast_kc_discovery.py BePKT --split-file folds/fold0_students.json
  python ast_kc_discovery.py Code-S --mode transform --artifact-dir kc_model \
      --input-jsonl new_submissions.jsonl --out new_attribution.json

Dependencies: numpy, pandas, torch, tree-sitter, tree-sitter-java,
tree-sitter-c, tree-sitter-cpp. No submitted code is executed or compiled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

SCHEMA = "dual-kc-1.0"
REASON_NAMES = ["logic", "syntax", "careless", "proficiency", "logicalError"]
REASON_WEIGHTS = [1.0, 1.0, 0.3, 0.3, 1.0]
CONCEPT_TYPES = {
    "for_statement", "enhanced_for_statement", "for_range_loop",
    "while_statement", "do_statement", "if_statement", "switch_statement",
    "switch_expression", "assignment_expression", "binary_expression",
    "unary_expression", "update_expression", "method_invocation",
    "call_expression", "array_creation_expression", "object_creation_expression",
    "array_access", "subscript_expression", "conditional_expression",
    "ternary_expression", "try_statement", "catch_clause", "throw_statement",
    "pointer_expression", "sizeof_expression", "cast_expression",
    "struct_specifier", "enum_specifier", "return_statement",
    "local_variable_declaration", "declaration", "lambda_expression",
}
LOOPS = {"for_statement", "enhanced_for_statement", "for_range_loop",
         "while_statement", "do_statement"}
CONTROLS = LOOPS | {"if_statement", "switch_statement", "switch_expression"}
PARSERS = {}


@dataclass
class Config:
    seed: int = 42
    max_depth: int = 3  # relative depth within EACH candidate subtree
    all_types: bool = False
    max_tokens: int = 96
    max_patterns: int = 48  # independently capped for each branch
    context_window: int = 8
    concept_dim: int = 64
    malformed_dim: int = 32
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.001
    reason_weight: float = 0.5
    concept_clusters: int = 40
    malformed_clusters_per_family: int = 4
    temperature_c: float = 0.15
    temperature_m: float = 0.15
    assignment_threshold: float = 0.15
    graph_neighbors: int = 5
    min_transition_support: int = 2
    transition_stability: int = 2


@dataclass
class Pattern:
    tokens: list[str]
    branch: str
    root_type: str
    family: str
    start_byte: int
    end_byte: int
    damaged: bool = False
    recovered: bool = False


@dataclass
class Submission:
    student_id: str
    problem_id: str
    code_id: str
    code: str
    timestamp: str | None = None
    language: str = "java"
    score_raw: float | None = None
    score: float | None = None
    reason: list[float | None] = field(default_factory=lambda: [None] * 5)
    compile_ok: bool | None = None
    compile_source: str = "unknown"
    split: str = ""
    event_id: str = ""
    # These orders come from the unfiltered canonical event master. Keep them
    # separate: KT order is useful for history construction, while only the
    # raw same-problem attempt order proves adjacency for temporal KC edges.
    original_event_order: int | None = None
    problem_attempt_order: int | None = None
    kt_sequence_order: int | None = None
    filtered_sequence_index: int | None = None
    order: int = 0
    normal_subtrees: list[Pattern] = field(default_factory=list)
    malformed_windows: list[Pattern] = field(default_factory=list)
    parse_error_count: int = 0


def finite_number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def present(value):
    return value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}


def first(row, *keys, default=None):
    for key in keys:
        if present(row.get(key)):
            return row[key]
    return default


def read_reason(value):
    if not present(value):
        return [None] * 5
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            try:
                value = ast.literal_eval(value)  # never eval student-controlled text
            except (ValueError, SyntaxError):
                return [None] * 5
    if not isinstance(value, (list, tuple)) or len(value) < 5:
        return [None] * 5
    result = [finite_number(v) for v in value[:5]]
    if any(v is not None and not 0 <= v <= 1 for v in result):
        raise ValueError("Reason labels must be probabilities in [0, 1].")
    return result


def compile_label(value, success_values=(), failure_values=()):
    """Decode explicit compile results. Numeric judge verdicts are NOT guessed."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    success = {"true", "success", "compile.success", "compiled", "compile_success"}
    failure = {"false", "error", "compile.error", "compilation error", "compile_error", "failed"}
    if text in success | {str(v).lower() for v in success_values}:
        return True
    if text in failure | {str(v).lower() for v in failure_values}:
        return False
    return None


def normalized_language(value, default):
    text = str(value or default).lower()
    if "java" in text:
        return "java"
    if "++" in text or "cpp" in text or "cxx" in text:
        return "cpp"
    if text in {"c", "gnu c", "c99", "c11", "c17"}:
        return "c"
    if text in {"", "nan", "unknown"}:
        return default
    raise ValueError(f"Unsupported language {value!r}; supply --language java/c/cpp.")


def parser_for(language):
    if language not in PARSERS:
        from tree_sitter import Language, Parser
        try:
            grammar = importlib.import_module(f"tree_sitter_{language}")
        except ImportError as exc:
            raise RuntimeError(f"Install tree-sitter-{language} for {language} submissions") from exc
        PARSERS[language] = Parser(Language(grammar.language()))
    return PARSERS[language]


def walk(node):
    stack = [node]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def ancestors(node):
    values = []
    node = node.parent
    while node is not None:
        values.append(node.type)
        node = node.parent
    return values


def lexical_token(node, raw):
    kind = node.type
    if "comment" in kind:
        return None
    if "identifier" in kind:
        return "<ID>"
    if "literal" in kind or kind in {"number", "true", "false", "null"}:
        return "<LIT>"
    if not node.children:
        text = raw[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        return text if text and len(text) <= 24 else kind
    return kind


def subtree_tokens(node, raw, max_depth, recovered_root=None):
    """Balanced structural serialization; truncation happens only at ID encoding."""
    tokens = []
    def visit(current, depth):
        if "comment" in current.type or (current.type == "ERROR" and not (depth == 0 and recovered_root)) or current.is_missing:
            return
        if "literal" in current.type:
            tokens.append("<LIT>")
            return
        if not current.children:
            token = lexical_token(current, raw)
            if token is not None:
                tokens.append(token)
            return
        tokens.append("node:" + (recovered_root if depth == 0 and recovered_root else current.type))
        if depth < max_depth:
            tokens.append("<CHILDREN>")
            for child in current.children:
                visit(child, depth + 1)
            tokens.append("</CHILDREN>")
    visit(node, 0)
    return tokens


def malformed_family(node, chain):
    if any(t in CONTROLS for t in chain) or any(c.type in {"for", "if", "while", "switch"} for c in node.children):
        return "Control"
    if any(t in {"formal_parameters", "parameter_list", "function_declarator"} for t in chain):
        return "Method"
    if any("declaration" in t or "declarator" in t for t in chain):
        return "Declaration"
    if node.is_missing or any("argument" in t for t in chain):
        return "Delimiter"
    return "Other"


def parse_submission(sub, config):
    parser = parser_for(sub.language)
    original = sub.code.encode("utf-8")
    raw, offset = original, 0
    root = parser.parse(raw).root_node
    # Java datasets sometimes contain a method rather than a complete class.
    if sub.language == "java" and root.has_error:
        prefix = b"class __KC_WRAPPER__ {\n"
        candidate = prefix + original + b"\n}"
        candidate_root = parser.parse(candidate).root_node
        errors = lambda r: sum(n.type == "ERROR" or n.is_missing for n in walk(r))
        if errors(candidate_root) < errors(root):
            raw, offset, root = candidate, len(prefix), candidate_root
    nodes = list(walk(root))
    leaves = [n for n in nodes if (not n.children or "literal" in n.type)
              and not n.is_missing and "comment" not in n.type
              and not any("literal" in t or "comment" in t for t in ancestors(n))]
    leaf_tokens = [lexical_token(n, raw) for n in leaves]
    sub.normal_subtrees, sub.malformed_windows = [], []
    seen_c, seen_m = set(), set()
    for node in nodes:
        if node.start_byte < offset or node.end_byte > offset + len(original):
            continue  # wrapper-only nodes must not become KCs
        chain = ancestors(node)
        parent = chain[0] if chain else "ROOT"
        if node.type == "ERROR" or node.is_missing:
            # A missing delimiter can collapse a whole loop into ERROR in CST.
            # Keep a marked structural candidate from an explicit leading keyword;
            # it may be attributed, but must NEVER define clean C prototypes.
            recovery_types = {"for": "for_statement", "if": "if_statement",
                              "while": "while_statement", "switch": "switch_statement"}
            leading = node.children[0].type if node.children else ""
            recovered = recovery_types.get(leading) if node.type == "ERROR" else None
            if recovered:
                c_tokens = ["root:"+recovered, "parent:"+parent]
                c_tokens += ["control:"+t for t in chain[:6] if t in CONTROLS]
                c_tokens += subtree_tokens(node, raw, config.max_depth, recovered)
                if tuple(c_tokens) not in seen_c:
                    sub.normal_subtrees.append(Pattern(c_tokens, "C", recovered, "Conceptual",
                        node.start_byte-offset, node.end_byte-offset, True, True))
                    seen_c.add(tuple(c_tokens))
            family = malformed_family(node, chain)
            before = [i for i, leaf in enumerate(leaves) if leaf.end_byte <= node.start_byte]
            after = [i for i, leaf in enumerate(leaves) if leaf.start_byte >= node.end_byte]
            inside = [i for i, leaf in enumerate(leaves)
                      if leaf.start_byte >= node.start_byte and leaf.end_byte <= node.end_byte]
            left = [leaf_tokens[i] for i in before[-config.context_window:]]
            right = [leaf_tokens[i] for i in after[:config.context_window]]
            center = [leaf_tokens[i] for i in inside[:config.context_window * 2]]
            window = [t for t in left + center + right if t is not None]
            prev = node.prev_sibling.type if node.prev_sibling else "NONE"
            nxt = node.next_sibling.type if node.next_sibling else "NONE"
            tokens = ["family:" + family, "error:MISSING" if node.is_missing else "error:ERROR",
                      "missing:" + node.type if node.is_missing else "missing:NONE",
                      "parent:" + parent, "grandparent:" + (chain[1] if len(chain) > 1 else "ROOT"),
                      "previous:" + prev, "next:" + nxt,
                      "relative_depth:" + str(min(len(chain), 12)),
                      "span_log2:" + str(int(math.log2(1 + node.end_byte - node.start_byte)))]
            tokens += ["inside:" + t for t in chain if t in CONTROLS or "declaration" in t
                       or t in {"formal_parameters", "parameter_list", "function_declarator"}]
            for opening, closing in [("(", ")"), ("[", "]"), ("{", "}")]:
                balance = max(-3, min(3, window.count(opening) - window.count(closing)))
                tokens.append(f"window_balance:{opening}{closing}:{balance}")
            tokens += ["<LEFT>"] + left + ["<ERROR_SPAN>"] + center + ["<RIGHT>"] + right
            tokens = [t for t in tokens if t is not None]
            key = (node.start_byte, node.end_byte, node.type)
            if key not in seen_m:
                sub.malformed_windows.append(Pattern(tokens, "M", node.type, family,
                                                      node.start_byte-offset, node.end_byte-offset, True))
                seen_m.add(key)
        elif node.type in CONCEPT_TYPES or (config.all_types and node.is_named and node.children):
            if "comment" in node.type:
                continue
            tokens = ["root:" + node.type, "parent:" + parent]
            tokens += ["control:" + t for t in chain[:6] if t in CONTROLS]
            descendants = list(walk(node))
            if node.type in LOOPS and any(n.type in LOOPS for n in descendants[1:]):
                tokens.append("nested_loop:true")
            if any(n.type in {"call_expression", "method_invocation"} for n in descendants):
                tokens.append("method_call:true")
            tokens += subtree_tokens(node, raw, config.max_depth)
            key = tuple(tokens)
            if key not in seen_c:
                sub.normal_subtrees.append(Pattern(tokens, "C", node.type, "Conceptual",
                                                   node.start_byte-offset, node.end_byte-offset, node.has_error))
                seen_c.add(key)
    sub.parse_error_count = len(sub.malformed_windows)
    # Whole-program traversal is never cut off by the relative subtree depth.
    # Evenly spaced selection avoids keeping only the first source-code prefix.
    for attr in ["normal_subtrees", "malformed_windows"]:
        patterns = getattr(sub, attr)
        if len(patterns) > config.max_patterns:
            chosen = np.linspace(0, len(patterns)-1, config.max_patterns, dtype=int)
            setattr(sub, attr, [patterns[i] for i in chosen])


def student_splits(students, seed, split_file=None):
    students = sorted(set(students))
    if split_file:
        data = json.loads(Path(split_file).read_text(encoding="utf-8-sig"))
        mapping = {}
        if any(k in data for k in ["train", "validation", "valid", "test"]):
            for split, members in data.items():
                split = "validation" if split == "valid" else split
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"Invalid split name: {split}")
                for sid in members:
                    sid = str(sid)
                    if sid in mapping:
                        raise ValueError(f"Student occurs in multiple splits: {sid}")
                    mapping[sid] = split
        else:
            mapping = {str(k): ("validation" if v == "valid" else v) for k, v in data.items()}
        if any(v not in {"train", "validation", "test"} for v in mapping.values()):
            raise ValueError("Splits must be train/validation/test.")
        missing = set(students) - mapping.keys()
        if missing:
            raise ValueError(f"Split manifest does not cover {len(missing)} students.")
        return mapping
    # Stable per-student hashing: adding new submissions cannot reshuffle students.
    return {sid: ("train" if (v := int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16)
                  / 2**32) < .8 else "validation" if v < .9 else "test") for sid in students}


def build_vocab(submissions, attr):
    tokens = {token for sub in submissions for p in getattr(sub, attr) for token in p.tokens}
    return {"<PAD>": 0, "<UNK>": 1, **{t: i+2 for i, t in enumerate(sorted(tokens - {"<PAD>", "<UNK>"}))}}


def pattern_ids(patterns, vocab, max_tokens):
    return [[vocab.get(t, 1) for t in p.tokens[:max_tokens]] for p in patterns]


def branch_tensor(submissions, attr, vocab, config, device):
    count = max(1, max((len(getattr(s, attr)) for s in submissions), default=0))
    ids = torch.zeros((len(submissions), count, config.max_tokens), dtype=torch.long, device=device)
    for b, sub in enumerate(submissions):
        for i, seq in enumerate(pattern_ids(getattr(sub, attr), vocab, config.max_tokens)):
            ids[b, i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return ids


class BranchEncoder(nn.Module):
    def __init__(self, vocab_size, dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.sequence = nn.GRU(dimension, dimension, batch_first=True)
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, 1))

    def forward(self, ids):
        batch, patterns, tokens = ids.shape
        flat = ids.reshape(-1, tokens)
        lengths = (flat != 0).sum(1)
        valid = lengths > 0
        z = self.embedding.weight.new_zeros((len(flat), self.embedding.embedding_dim))
        if valid.any():
            embedded = self.embedding(flat[valid])
            packed = pack_padded_sequence(embedded, lengths[valid].cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.sequence(packed)
            encoded = F.normalize(self.norm(hidden[-1]), dim=-1)
            z = z.index_copy(0, valid.nonzero().flatten(), encoded)
        z = z.reshape(batch, patterns, -1)
        mask = valid.reshape(batch, patterns)
        logits = self.attention(z).squeeze(-1).masked_fill(~mask, -1e9)
        weights = logits.softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (z * weights.unsqueeze(-1)).sum(1)
        return z, weights, pooled


class DualKCModel(nn.Module):
    def __init__(self, vocab_c, vocab_m, config):
        super().__init__()
        self.concept = BranchEncoder(len(vocab_c), config.concept_dim)
        self.malformed = BranchEncoder(len(vocab_m), config.malformed_dim)
        self.score_head = nn.Linear(config.concept_dim, 1)
        self.logic_head = nn.Linear(config.concept_dim, 2)
        self.syntax_head = nn.Linear(config.malformed_dim, 1)
        self.compile_head = nn.Linear(config.malformed_dim, 1)
        self.reason_head = nn.Linear(config.concept_dim + config.malformed_dim, 5)

    def forward(self, c_ids, m_ids):
        zc, ac, hc = self.concept(c_ids)
        zm, am, hm = self.malformed(m_ids)
        return {"zc": zc, "ac": ac, "zm": zm, "am": am,
                "score": self.score_head(hc).squeeze(-1).sigmoid(),
                "logic": self.logic_head(hc), "syntax": self.syntax_head(hm).squeeze(-1),
                "compile": self.compile_head(hm).squeeze(-1),
                "reason": self.reason_head(torch.cat([hc, hm], dim=-1))}


def masked_loss(pred, labels, binary=False, weights=None):
    target = torch.tensor(labels, dtype=pred.dtype, device=pred.device)
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0, 0
    safe = torch.nan_to_num(target)
    values = F.binary_cross_entropy_with_logits(pred, safe, reduction="none") if binary else (pred-safe).square()
    if weights is not None:
        values = values * torch.tensor(weights, dtype=pred.dtype, device=pred.device)
    return values[mask].mean(), int(mask.sum())


def train_model(model, train, vocab_c, vocab_m, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = random.Random(config.seed)
    number = lambda v: float("nan") if v is None else float(v)
    history = []
    for epoch in range(config.epochs):
        model.train()
        indices = list(range(len(train)))
        rng.shuffle(indices)
        losses, supervised = [], 0
        for start in range(0, len(indices), config.batch_size):
            batch = [train[i] for i in indices[start:start+config.batch_size]]
            output = model(branch_tensor(batch, "normal_subtrees", vocab_c, config, device),
                           branch_tensor(batch, "malformed_windows", vocab_m, config, device))
            score, ns = masked_loss(output["score"], [number(s.score) for s in batch])
            logic, nl = masked_loss(output["logic"], [[number(s.reason[0]), number(s.reason[4])] for s in batch], True)
            syntax, ny = masked_loss(output["syntax"], [number(s.reason[1]) for s in batch], True)
            comp, nc = masked_loss(output["compile"], [number(s.compile_ok) for s in batch], True)
            reason, nr = masked_loss(output["reason"], [[number(v) for v in s.reason] for s in batch], True, REASON_WEIGHTS)
            loss = score + logic + syntax + comp + config.reason_weight * reason
            if ns + nl + ny + nc + nr == 0:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; inspect labels and patterns.")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised += ns + nl + ny + nc + nr
        if supervised == 0:
            raise ValueError("No observed training labels. Missing labels are not converted to negatives.")
        history.append({"epoch": epoch+1, "loss": float(np.mean(losses)), "observed_targets": supervised})
        print(f"Epoch {epoch+1}/{config.epochs}: loss={history[-1]['loss']:.5f}")
    model.eval()
    return history


def weighted_kmeans(vectors, weights, n_clusters, seed=42, max_iter=100):
    """Minimize sum_i w_i ||x_i - center[label_i]||^2; never rescale x_i."""
    x = np.asarray(vectors, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or w.shape != (len(x),):
        raise ValueError("Expected nonempty [N,D] vectors and [N] weights.")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Clustering requires finite vectors and strictly positive weights.")
    unique = len(np.unique(x, axis=0))
    k = min(n_clusters, unique)
    if k < 1:
        raise ValueError("At least one cluster is required.")
    rng = np.random.default_rng(seed)
    centers = [x[rng.choice(len(x), p=w/w.sum())].copy()]
    for _ in range(1, k):
        distance = np.min(((x[:, None] - np.asarray(centers)[None])**2).sum(-1), axis=1)
        mass = w * distance
        centers.append(x[rng.choice(len(x), p=mass/mass.sum())].copy())
    centers = np.asarray(centers)
    for _ in range(max_iter):
        distances = ((x[:, None] - centers[None])**2).sum(-1)
        labels = distances.argmin(1)
        updated = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                updated[c] = np.average(x[mask], weights=w[mask], axis=0)
            else:
                updated[c] = x[np.argmax(w * distances.min(1))]
        if np.allclose(centers, updated, atol=1e-7, rtol=1e-6):
            centers = updated
            break
        centers = updated
    labels = ((x[:, None] - centers[None])**2).sum(-1).argmin(1)
    # Remove empty centers and compact labels; exported IDs must index matrices.
    used = np.unique(labels)
    remap = {int(old): new for new, old in enumerate(used)}
    return np.array([remap[int(v)] for v in labels]), centers[used].astype(np.float32)


def conceptual_prototype(sub):
    logic, syntax, _, _, error = sub.reason
    if sub.compile_ok is False or (error is not None and error >= .5):
        return False
    if logic is not None and syntax is not None:
        return logic >= .5 and syntax >= .5
    # Explicit full success plus observed compilation is a documented fallback.
    return sub.compile_ok is True and sub.score is not None and sub.score >= 1.0


def malformed_prototype(sub):
    """仅由真实判题器确认的编译失败提交定义 M-KC 原型。"""
    return sub.compile_ok is False


@torch.no_grad()
def representations(model, submissions, vocab_c, vocab_m, config, device):
    model.eval()
    for start in range(0, len(submissions), config.batch_size):
        batch = submissions[start:start+config.batch_size]
        output = model(branch_tensor(batch, "normal_subtrees", vocab_c, config, device),
                       branch_tensor(batch, "malformed_windows", vocab_m, config, device))
        for i, sub in enumerate(batch):
            nc, nm = len(sub.normal_subtrees), len(sub.malformed_windows)
            yield sub, output["zc"][i, :nc].cpu().numpy(), output["ac"][i, :nc].cpu().numpy(), \
                output["zm"][i, :nm].cpu().numpy(), output["am"][i, :nm].cpu().numpy()


def fit_clusters(model, train, vocab_c, vocab_m, config, device):
    samples = {"C": {}, "M": {}}
    for sub, zc, ac, zm, am in representations(model, train, vocab_c, vocab_m, config, device):
        for branch, patterns, zs, attention, eligible in [
            ("C", sub.normal_subtrees, zc, ac, conceptual_prototype(sub)),
            ("M", sub.malformed_windows, zm, am, malformed_prototype(sub)),
        ]:
            if not eligible:
                continue
            for pattern, vector, weight in zip(patterns, zs, attention):
                if branch == "C" and pattern.damaged:
                    continue
                # IDs, vectors, weights, and exemplars stay in ONE record.
                key = (pattern.family, tuple(pattern.tokens[:config.max_tokens]))
                sample = samples[branch].setdefault(key, {"vector": vector, "weight": 0., "count": 0,
                    "pattern": pattern, "examples": [], "students": set()})
                sample["weight"] += max(float(weight), 1e-8)
                sample["count"] += 1
                sample["students"].add(sub.student_id)
                if len(sample["examples"]) < 3:
                    sample["examples"].append({"code_id": sub.code_id, "start_byte": pattern.start_byte,
                                               "end_byte": pattern.end_byte})
    if not samples["C"]:
        raise ValueError("No reliable Conceptual prototypes in train. Need logic=1 and syntax=1, "
                         "or observed compile success + full score; inspect labels/parse coverage.")
    result = {}
    for branch, dimension in [("C", config.concept_dim), ("M", config.malformed_dim)]:
        by_family = defaultdict(list)
        for sample in samples[branch].values():
            by_family[sample["pattern"].family].append(sample)
        centers, patterns, families = [], {}, []
        for family, members in sorted(by_family.items()):
            x = np.stack([s["vector"] for s in members])
            weights = np.array([s["weight"] for s in members])
            requested = config.concept_clusters if branch == "C" else config.malformed_clusters_per_family
            labels, family_centers = weighted_kmeans(x, weights, requested, config.seed)
            for local, center in enumerate(family_centers):
                kid = len(centers)
                selected = [s for s, label in zip(members, labels) if label == local]
                representative = min(selected, key=lambda s: float(np.linalg.norm(s["vector"]-center)))
                roots = Counter({})
                for sample in selected:
                    roots[sample["pattern"].root_type] += sample["count"]
                patterns[str(kid)] = {
                    "id": f"{branch}{kid:03d}", "family": family,
                    "description": family + ":" + representative["pattern"].root_type,
                    "num_members": sum(s["count"] for s in selected),
                    "unique_patterns": len(selected),
                    "student_support": len(set().union(*(s["students"] for s in selected))),
                    "type_distribution": dict(roots),
                    "representative_tokens": representative["pattern"].tokens[:config.max_tokens],
                    "examples": representative["examples"],
                }
                centers.append(center.tolist())
                families.append(family)
        result[branch] = {"num_kcs": len(centers), "dimension": dimension, "centers": centers,
                          "families": families, "patterns": patterns}
    return result


def distance_softmax(vector, centers, temperature):
    if len(centers) == 0:
        return np.zeros(0, dtype=np.float32)
    logits = -np.linalg.norm(np.asarray(centers)-vector, axis=1) / temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def assign_branch(patterns, vectors, attention, cluster, temperature, threshold):
    count = cluster["num_kcs"]
    probabilities = np.zeros(count, dtype=np.float64)
    weighted_mass = np.zeros(count, dtype=np.float64)
    evidence = defaultdict(list)
    for pattern, vector, weight in zip(patterns, vectors, attention):
        eligible = [k for k, family in enumerate(cluster["families"])
                    if pattern.branch == "C" or family == pattern.family]
        if not eligible:
            continue
        soft = distance_softmax(vector, [cluster["centers"][k] for k in eligible], temperature)
        for k, probability in zip(eligible, soft):
            probabilities[k] = max(probabilities[k], float(probability))
            weighted_mass[k] += float(weight) * float(probability)
            if probability > threshold:
                evidence[k].append({"start_byte": pattern.start_byte, "end_byte": pattern.end_byte,
                                    "root_type": pattern.root_type, "family": pattern.family,
                                    "recovered_from_error": pattern.recovered,
                                    "probability": float(probability), "attention": float(weight)})
    selected = [k for k in range(count) if probabilities[k] > threshold]
    return {"ids": selected, "weights": {str(k): float(probabilities[k]) for k in selected},
            "attention_mass": {str(k): float(weighted_mass[k]) for k in selected},
            "evidence": {str(k): sorted(evidence[k], key=lambda e: -e["probability"])[:3] for k in selected}}


def attribute(model, submissions, vocab_c, vocab_m, clusters, config, device):
    events = {}
    for sub, zc, ac, zm, am in representations(model, submissions, vocab_c, vocab_m, config, device):
        c = assign_branch(sub.normal_subtrees, zc, ac, clusters["C"], config.temperature_c, config.assignment_threshold)
        m = assign_branch(sub.malformed_windows, zm, am, clusters["M"], config.temperature_m, config.assignment_threshold)
        state = "observed_structure"
        if sub.reason[1] is not None and sub.reason[1] >= .5 and sub.reason[4] is not None and sub.reason[4] >= .5:
            state = "incorrect_implementation_evidence"
        elif conceptual_prototype(sub):
            state = "correct_submission_evidence"
        events[sub.event_id] = {
            "code_id": sub.code_id, "student_id": sub.student_id, "problem_id": sub.problem_id,
            "timestamp": sub.timestamp, "split": sub.split,
            "original_event_order": sub.original_event_order,
            "problem_attempt_order": sub.problem_attempt_order,
            "kt_sequence_order": sub.kt_sequence_order,
            "filtered_sequence_index": sub.filtered_sequence_index,
            "concept": c["ids"], "malformed": m["ids"],
            "concept_kcs": {f"C{int(k):03d}": v for k, v in c["weights"].items()},
            "malformed_kcs": {f"M{int(k):03d}": v for k, v in m["weights"].items()},
            "kc_mask": c["ids"] + [clusters["C"]["num_kcs"]+k for k in m["ids"]],
            "concept_attribution": c, "malformed_attribution": m,
            "concept_implementation_state": state, "state_scope": "submission_weak_evidence_not_per_kc_ground_truth",
            "score": sub.score, "score_raw": sub.score_raw, "reason": sub.reason,
            "compile_ok": sub.compile_ok, "compile_source": sub.compile_source,
            "parse_error_count": sub.parse_error_count,
            "unlocalized_compile_failure": sub.compile_ok is False and sub.parse_error_count == 0,
            "unmodeled_malformed_families": sorted({p.family for p in sub.malformed_windows} - set(clusters["M"]["families"])),
        }
    return events


def cosine_matrix(vectors, count):
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    normed = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    result = np.clip(normed @ normed.T, 0., 1.)
    np.fill_diagonal(result, 0.)
    return result


def sparsify(matrix, neighbors, symmetric=False):
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    n = len(matrix)
    result = np.zeros_like(matrix)
    if n < 2 or neighbors <= 0:
        return result
    np.fill_diagonal(matrix, 0.)
    for i in range(n):
        positive = np.flatnonzero(matrix[i] > 0)
        chosen = positive[np.argsort(-matrix[i, positive], kind="stable")[:min(neighbors, n-1)]]
        result[i, chosen] = matrix[i, chosen]
    return np.maximum(result, result.T) if symmetric else result


def build_graph(events, clusters, config):
    """Build a training-only graph using consecutive raw same-problem attempts.

    Different-problem submissions may occur between the two attempts. Filtered
    KT order is retained for history construction and never establishes graph
    adjacency. Rows are sources, columns targets; MC is M -> C.
    """
    rows = [e for e in events.values() if e["split"] == "train"]
    nc, nm = clusters["C"]["num_kcs"], clusters["M"]["num_kcs"]
    q = np.zeros((len(rows), nc))
    for i, event in enumerate(rows):
        q[i, event["concept"]] = 1.
    co = q.T @ q
    denom = np.sqrt(np.diag(co)[:, None] * np.diag(co)[None, :])
    co = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
    np.fill_diagonal(co, 0.)
    cc = sparsify(.5*cosine_matrix(clusters["C"]["centers"], nc)+.5*co, config.graph_neighbors, True)
    mm_structure = cosine_matrix(clusters["M"]["centers"], nm)
    for i in range(nm):
        for j in range(nm):
            if clusters["M"]["families"][i] != clusters["M"]["families"][j]:
                mm_structure[i, j] = 0.
    transitions = np.zeros((nm, nm))
    mc_counts, resolved = np.zeros((nm, nc)), np.zeros(nm)
    baseline, eligible_windows, skipped = np.zeros(nc), 0, 0
    eligible_pairs, rejected_order_gap = 0, 0
    original_order_gap_windows, global_direct_windows = 0, 0
    groups = defaultdict(list)
    for event in rows:
        stamp = pd.to_datetime(event["timestamp"], errors="coerce", utc=True)
        if pd.isna(stamp):
            skipped += 1
            continue
        groups[(event["student_id"], event["problem_id"])].append((stamp, event))
    for group in groups.values():
        group.sort(key=lambda pair: (pair[1]["problem_attempt_order"] is None,
                                     pair[1]["problem_attempt_order"] if pair[1]["problem_attempt_order"] is not None else 0,
                                     pair[0]))
        for t in range(len(group)-1):
            time_a, current = group[t]
            time_b, nxt = group[t+1]
            order_a, order_b = current["problem_attempt_order"], nxt["problem_attempt_order"]
            if order_a is None or order_b is None or order_b != order_a + 1:
                rejected_order_gap += 1
                continue  # never bridge a filtered same-problem attempt
            if time_b <= time_a:
                skipped += 1
                continue
            eligible_pairs += 1
            if (current["original_event_order"] is not None and nxt["original_event_order"] is not None):
                if nxt["original_event_order"] != current["original_event_order"] + 1:
                    original_order_gap_windows += 1
                else:
                    global_direct_windows += 1
            for i in current["malformed"]:
                for j in nxt["malformed"]:
                    transitions[i, j] += 1.
            future = group[t+1:t+1+config.transition_stability]
            if len(future) != config.transition_stability:
                continue
            if any(e["compile_ok"] is not True for _, e in future):
                continue
            if any(e["problem_attempt_order"] is None or
                   e["problem_attempt_order"] != order_a+j+1 for j, (_, e) in enumerate(future)):
                continue
            if any(future[j][0] <= future[j-1][0] for j in range(1, len(future))):
                continue
            if all(current["original_event_order"] is not None and e["original_event_order"] is not None
                   for _, e in future):
                orders = [current["original_event_order"]] + [e["original_event_order"] for _, e in future]
                if any(orders[j] != orders[j-1] + 1 for j in range(1, len(orders))):
                    original_order_gap_windows += 1
                else:
                    global_direct_windows += 1
            stable_c = set.intersection(*(set(e["concept"]) for _, e in future))
            eligible_windows += 1
            for j in stable_c:
                baseline[j] += 1.
            if current["compile_ok"] is not False:
                continue
            persistent_m = set.union(*(set(e["malformed"]) for _, e in future))
            for i in set(current["malformed"]) - persistent_m:
                resolved[i] += 1.
                for j in stable_c:
                    mc_counts[i, j] += 1.
    mm_counts = transitions.copy()
    transitions[transitions < config.min_transition_support] = 0.
    np.fill_diagonal(transitions, 0.)
    transitions /= np.maximum(transitions.sum(1, keepdims=True), 1.)
    mm = sparsify(.5*mm_structure + .5*transitions, config.graph_neighbors, False)
    probability = np.divide(mc_counts, resolved[:, None], out=np.zeros_like(mc_counts), where=resolved[:, None] > 0)
    base_prob = baseline/max(eligible_windows, 1)
    lift = np.divide(probability, base_prob[None, :], out=np.zeros_like(probability), where=base_prob[None, :] > 0)
    mc = probability * np.maximum(np.log(np.maximum(lift, 1.)), 0.)
    mc[mc_counts < config.min_transition_support] = 0.
    cm = np.zeros((nc, nm))  # no fabricated reverse/cross-space cosine edges
    adjacency = np.block([[cc, cm], [mc, mm]])
    np.fill_diagonal(adjacency, 1.)
    return {"CC": cc.tolist(), "CM": cm.tolist(), "MC": mc.tolist(), "MM": mm.tolist(),
            "MM_structure": mm_structure.tolist(), "MM_transition_counts": mm_counts.tolist(),
            "MC_support": mc_counts.tolist(), "MC_resolved_counts": resolved.tolist(),
            "MC_probability": probability.tolist(), "MC_lift": lift.tolist(),
            "MC_baseline_probability": base_prob.tolist(), "eligible_temporal_windows": eligible_windows,
            "eligible_temporal_pairs": eligible_pairs,
            "rejected_nonconsecutive_problem_attempt_pairs": rejected_order_gap,
            "original_order_gap_temporal_windows": original_order_gap_windows,
            "globally_direct_temporal_windows": global_direct_windows,
            "skipped_temporal_rows_or_pairs": skipped, "adjacency": adjacency.tolist(),
            "node_order": [f"C{i:03d}" for i in range(nc)] + [f"M{i:03d}" for i in range(nm)],
            "orientation": "row_source_column_target", "fit_split": "train",
            "temporal_adjacency_policy": "consecutive_raw_same_problem_attempts; other-problem events may intervene; kt_sequence_order is not adjacency evidence",
            "temporal_order_fields": ["problem_attempt_order", "original_event_order", "kt_sequence_order", "filtered_sequence_index"],
            "MC_interpretation": "observed obstacle resolution followed by stable conceptual structure; not causal prerequisite",
            "stability_submissions": config.transition_stability}


def locate(directory, candidates, required=False):
    for candidate in candidates:
        path = directory / candidate
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(f"None of {candidates} found in {directory}")
    return None


def read_lookup(path, value_column):
    if path is None:
        return {}
    frame = pd.read_csv(path, dtype=str, usecols=["CodeStateID", value_column])
    if frame["CodeStateID"].duplicated().any():
        conflicts = frame.groupby("CodeStateID")[value_column].nunique()
        if (conflicts > 1).any():
            raise ValueError(f"Conflicting per-code labels in {path}; resolve before fitting.")
    return dict(zip(frame["CodeStateID"], frame[value_column]))


def load_submissions(args):
    """Metadata first, then selected code payloads; preserve unknown labels."""
    default_language = "cpp" if args.dataset == "BePKT" else "java"
    code_path = None
    compile_by_code = {}
    if args.input_jsonl:
        records = []
        with Path(args.input_jsonl).open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
        scores, reasons = {}, {}
    else:
        directory = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parents[2] / "data" / args.dataset
        code_path = Path(args.code_file) if args.code_file else locate(directory, ["CodeStates_5000.csv", "CodeStates.csv"], True)
        main_candidates = ["submission.csv", "MainTable.csv"] if args.dataset == "BePKT" else ["MainTable_5000.csv", "MainTable.csv"]
        main_path = Path(args.main_file) if args.main_file else locate(directory, main_candidates, True)
        header = pd.read_csv(main_path, nrows=0).columns
        allowed = {"CodeStateID", "code_id", "id", "ProblemID", "problem_id", "SubjectID", "student_id", "user_id",
                   "ServerTimestamp", "timestamp", "create_time", "Language", "language", "EventType", "EventID", "Order",
                   "Score", "score", "score_raw", "Reason", "reason", "Compile.Result", "compile_ok", "Result", "result"}
        if args.compile_column:
            allowed.add(args.compile_column)
        frame = pd.read_csv(main_path, dtype=str, usecols=[c for c in header if c in allowed])
        if "EventType" in frame and "Compile.Result" in frame:
            compile_rows = frame[frame["EventType"] == "Compile"]
            for row in compile_rows.to_dict("records"):
                key = (str(first(row, "SubjectID", "student_id", "user_id")),
                       str(first(row, "ProblemID", "problem_id")), str(first(row, "CodeStateID", "code_id", "id")))
                label = compile_label(row.get("Compile.Result"), args.compile_success_values, args.compile_failure_values)
                if label is not None:
                    compile_by_code.setdefault(key, set()).add(label)
            # Retain failed compilation attempts that never produced a Run.Program.
            run = frame[frame["EventType"] == "Run.Program"]
            run_keys = set(zip(run["SubjectID"], run["ProblemID"], run["CodeStateID"]))
            compile_only = compile_rows[[tuple(row) not in run_keys for row in
                                         compile_rows[["SubjectID", "ProblemID", "CodeStateID"]].itertuples(index=False, name=None)]]
            frame = pd.concat([run, compile_only], ignore_index=True)
        records = frame.to_dict("records")
        score_path = Path(args.score_file) if args.score_file else locate(directory, ["CodeScore.csv", "V30/CodeScore_5000.csv", "V30/CodeScore.csv"])
        reason_path = Path(args.reason_file) if args.reason_file else locate(directory, ["CodeReason.csv", "V30/CodeReason_5000.csv", "V30/CodeReason.csv"])
        scores, reasons = read_lookup(score_path, "Score"), read_lookup(reason_path, "Reason")
    submissions, seen_events = [], set()
    for row_index, row in enumerate(records):
        sid = first(row, "student_id", "SubjectID", "user_id")
        pid = first(row, "problem_id", "ProblemID")
        cid = first(row, "code_id", "CodeStateID", "id")
        if any(not present(v) for v in (sid, pid, cid)):
            continue
        sid, pid, cid = str(sid), str(pid), str(cid)
        timestamp = first(row, "timestamp", "ServerTimestamp", "create_time")
        event_key = (sid, pid, cid, str(timestamp), str(first(row, "EventID", "event_id", default="")))
        if event_key in seen_events:
            continue
        seen_events.add(event_key)
        event_id = str(first(row, "event_id", "EventID", default=""))
        if not event_id:
            event_id = "event_" + hashlib.sha256(json.dumps(event_key).encode()).hexdigest()[:20]
        raw = finite_number(first(row, "score_raw"))
        normalized = finite_number(first(row, "score"))
        # JSONL score means [0,1]; external CodeScore.csv is explicitly [0, score_max].
        if cid in scores:
            raw = finite_number(scores[cid])
            normalized = None if raw is None else raw/args.score_max
        elif raw is not None:
            normalized = raw/args.score_max
        elif normalized is None:
            original_score = finite_number(row.get("Score"))
            if original_score is not None:
                normalized = original_score/args.main_score_max
                raw = normalized*args.score_max
        if normalized is not None and not 0 <= normalized <= 1:
            raise ValueError(f"Out-of-range score for {cid}; set score scales explicitly.")
        if raw is None and normalized is not None:
            raw = normalized*args.score_max
        reason = read_reason(reasons.get(cid, first(row, "reason", "Reason")))
        compile_column = args.compile_column or ("compile_ok" if "compile_ok" in row else "Compile.Result")
        compiled = compile_label(row.get(compile_column), args.compile_success_values, args.compile_failure_values)
        if compile_column == "compile_ok" and finite_number(row.get(compile_column)) in (0., 1.):
            compiled = bool(float(row[compile_column]))
        source = compile_column if compiled is not None else "unknown"
        values = compile_by_code.get((sid, pid, cid), set())
        if compiled is None and len(values) == 1:
            compiled, source = next(iter(values)), "Compile.Result"
        try:
            language = args.language or normalized_language(first(row, "language", "Language"), default_language)
        except ValueError:
            language = "unsupported:" + str(first(row, "language", "Language", default="unknown"))
        def order_value(*names):
            value = finite_number(first(row, *names))
            return int(value) if value is not None and value.is_integer() and value >= 0 else None
        submissions.append(Submission(sid, pid, cid, str(first(row, "code", "Code", default="")),
            timestamp=str(timestamp) if present(timestamp) else None,
            language=language,
            score_raw=raw, score=normalized, reason=reason, compile_ok=compiled, compile_source=source,
            event_id=event_id,
            original_event_order=order_value("original_event_order"),
            problem_attempt_order=order_value("problem_attempt_order"),
            kt_sequence_order=order_value("kt_sequence_order"),
            order=row_index))
    if not submissions:
        raise ValueError("No submissions with student_id, problem_id and code_id found.")
    if args.mode == "transform":
        saved = json.loads((Path(args.artifact_dir)/"metadata.json").read_text(encoding="utf-8"))
        splits = saved["student_splits"]
    else:
        splits = student_splits([s.student_id for s in submissions], args.seed, args.split_file)
    groups = defaultdict(list)
    for sub in submissions:
        sub.split = splits.get(sub.student_id, "inference")
        groups[(sub.student_id, sub.problem_id)].append(sub)
    for group in groups.values():
        def key(s):
            timestamp = pd.to_datetime(s.timestamp, errors="coerce", utc=True)
            raw_order = s.original_event_order
            return (raw_order is None, raw_order if raw_order is not None else
                    (timestamp.value if not pd.isna(timestamp) else s.order), s.order)
        group.sort(key=key)
        for index, sub in enumerate(group):
            # Infer same-problem order only for legacy inputs with no canonical
            # order fields. Such inferred values cannot close gaps left by a
            # language filter, so fold workflows must provide the event master.
            if sub.problem_attempt_order is None:
                sub.problem_attempt_order = index
            if sub.kt_sequence_order is None:
                sub.kt_sequence_order = index
            sub.order = sub.problem_attempt_order
    unsupported = Counter(s.language for s in submissions if s.language.startswith("unsupported:"))
    if unsupported:
        print(f"Warning: excluding unsupported languages (never parsed as C/C++): {dict(unsupported)}")
        submissions = [s for s in submissions if not s.language.startswith("unsupported:")]
    if not submissions:
        raise ValueError("No submissions in supported languages java/c/cpp.")
    # This diagnostic index describes only the retained language stream. It is
    # never used to infer raw same-problem adjacency.
    by_student = defaultdict(list)
    for sub in submissions:
        by_student[sub.student_id].append(sub)
    for group in by_student.values():
        group.sort(key=lambda s: (s.kt_sequence_order is None,
                                 s.kt_sequence_order if s.kt_sequence_order is not None else s.order,
                                 s.order))
        for index, sub in enumerate(group):
            sub.filtered_sequence_index = index
    limit = args.max_samples if args.max_samples is not None else (20000 if args.dataset == "BePKT" else 0)
    if limit and len(submissions) > limit:
        chosen = np.sort(np.random.default_rng(args.seed).choice(len(submissions), limit, replace=False))
        submissions = [submissions[i] for i in chosen]
    if code_path:
        wanted = {s.code_id for s in submissions}
        codes = {}
        for chunk in pd.read_csv(code_path, dtype=str, usecols=["CodeStateID", "Code"], chunksize=50000):
            for cid, code in chunk[chunk["CodeStateID"].isin(wanted)].itertuples(index=False, name=None):
                if present(code):
                    if cid in codes and codes[cid] != code:
                        raise ValueError(f"CodeStateID refers to multiple different sources: {cid}")
                    codes[cid] = code
        for sub in submissions:
            sub.code = codes.get(sub.code_id, "")
        missing = sum(not s.code for s in submissions)
        if missing:
            print(f"Warning: skipping {missing} submissions without code; sequence gaps are preserved.")
        submissions = [s for s in submissions if s.code]
    # A code-state key must identify one structural input, even across students.
    code_inputs = {}
    for sub in submissions:
        fingerprint = (sub.code, sub.language)
        if sub.code_id in code_inputs and code_inputs[sub.code_id] != fingerprint:
            raise ValueError(f"Conflicting source/language for CodeStateID {sub.code_id}")
        code_inputs[sub.code_id] = fingerprint
    return submissions, splits


def problem_summary(events):
    counts = defaultdict(Counter)
    totals = Counter()
    for event in events.values():
        if event["split"] != "train":
            continue
        pid = event["problem_id"]
        totals[pid] += 1
        counts[pid].update(event["kc_mask"])
    mapping = {pid: sorted(counter) for pid, counter in counts.items()}
    prevalence = {pid: {str(k): v/totals[pid] for k, v in counter.items()} for pid, counter in counts.items()}
    return mapping, prevalence


def code_state_map(events):
    grouped = defaultdict(list)
    for event_id, event in events.items():
        grouped[event["code_id"]].append((event_id, event))
    result = {}
    for cid, occurrences in grouped.items():
        # Attribution is code-only; event-specific labels must never be overwritten silently.
        first_event = occurrences[0][1]
        entry = {k: first_event[k] for k in ["concept", "malformed", "concept_kcs", "malformed_kcs", "kc_mask",
                 "concept_attribution", "malformed_attribution", "parse_error_count", "unmodeled_malformed_families"]}
        entry["event_ids"] = [key for key, _ in occurrences]
        entry["has_multiple_events"] = len(occurrences) > 1
        conflicts = []
        for name in ["score", "score_raw", "reason", "compile_ok", "compile_source", "concept_implementation_state"]:
            values = [event[name] for _, event in occurrences]
            if all(value == values[0] for value in values):
                entry[name] = values[0]
            else:
                entry[name] = None
                conflicts.append(name)
        entry["event_specific_fields"] = conflicts
        entry["state_scope"] = "submission_weak_evidence_not_per_kc_ground_truth"
        result[cid] = entry
    return result


def export_mapping(events, metadata):
    clusters = metadata["clusters"]
    nc, nm = clusters["C"]["num_kcs"], clusters["M"]["num_kcs"]
    descriptions = [clusters[b]["patterns"][str(i)]["description"] for b, n in [("C", nc), ("M", nm)] for i in range(n)]
    code_patterns = {str(i): clusters["C"]["patterns"][str(i)] for i in range(nc)}
    code_patterns.update({str(nc+i): clusters["M"]["patterns"][str(i)] for i in range(nm)})
    return {"schema_version": SCHEMA, "concept_kcs": clusters["C"], "malformed_kcs": clusters["M"],
            "submission_kc_map": code_state_map(events), "submission_events": events,
            "problem_kc_map": metadata["problem_kc_map"], "problem_kc_prevalence": metadata["problem_kc_prevalence"],
            "problem_kc_map_scope": "training-only auxiliary statistics, not submission attribution",
            "graph": metadata["graph"], "student_splits": metadata["student_splits"],
            "num_kcs": nc+nm, "kc_ids": list(range(nc+nm)), "kc_list": descriptions,
            "kc_code_patterns": code_patterns, "kc_adjacency": metadata["graph"]["adjacency"],
            "kc_types": ["Conceptual"]*nc + ["Malformed"]*nm,
            "global_index": {**{f"C{i:03d}": i for i in range(nc)}, **{f"M{i:03d}": nc+i for i in range(nm)}},
            "attribution_semantics": "max pattern soft-assignment; not calibrated mastery probability",
            "fit_summary": metadata["fit_summary"], "config": metadata["config"]}


def json_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_pipeline(submissions, splits, config, artifact_dir, mode="fit-transform", device="cpu"):
    artifact_dir = Path(artifact_dir)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if mode == "transform":
        metadata = json.loads((artifact_dir/"metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != SCHEMA:
            raise ValueError("Incompatible discovery artifact schema.")
        config = Config(**metadata["config"])
        for sub in submissions:
            sub.split = metadata["student_splits"].get(sub.student_id, "inference")
    for sub in submissions:
        parse_submission(sub, config)
    if mode == "transform":
        vocab_c, vocab_m = metadata["vocab_c"], metadata["vocab_m"]
        model = DualKCModel(vocab_c, vocab_m, config).to(device)
        model.load_state_dict(torch.load(artifact_dir/"weights.pt", map_location=device, weights_only=True))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        events = attribute(model, submissions, vocab_c, vocab_m, metadata["clusters"], config, device)
        # Transform mode freezes encoders/prototypes but must still rebuild
        # temporal graph evidence from the corrected raw order fields.
        metadata = dict(metadata)
        metadata["graph"] = build_graph(events, metadata["clusters"], config)
    else:
        for sub in submissions:
            if sub.student_id not in splits:
                raise ValueError(f"Missing split for student {sub.student_id}")
            sub.split = splits[sub.student_id]
        train = [s for s in submissions if s.split == "train"]
        if not train:
            raise ValueError("No training submissions. Supply a student split manifest or increase sample size.")
        if not any(conceptual_prototype(s) and any(not p.damaged for p in s.normal_subtrees) for s in train):
            raise ValueError("No clean Conceptual prototypes in training data; inspect reason/compile/score coverage.")
        vocab_c = build_vocab(train, "normal_subtrees")
        vocab_m = build_vocab(train, "malformed_windows")
        model = DualKCModel(vocab_c, vocab_m, config).to(device)
        history = train_model(model, train, vocab_c, vocab_m, config, device)
        clusters = fit_clusters(model, train, vocab_c, vocab_m, config, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        events = attribute(model, submissions, vocab_c, vocab_m, clusters, config, device)
        graph = build_graph(events, clusters, config)
        mapping, prevalence = problem_summary(events)
        training_content = sorted((s.student_id, s.problem_id, s.code_id,
                                   hashlib.sha256(s.code.encode()).hexdigest()) for s in train)
        metadata = {"schema_version": SCHEMA, "config": asdict(config), "vocab_c": vocab_c, "vocab_m": vocab_m,
            "clusters": clusters, "student_splits": splits, "graph": graph, "problem_kc_map": mapping,
            "problem_kc_prevalence": prevalence, "training_history": history,
            "fit_summary": {"train_submissions": len(train), "train_students": len({s.student_id for s in train}),
                "conceptual_prototype_submissions": sum(conceptual_prototype(s) for s in train),
                "malformed_prototype_submissions": sum(malformed_prototype(s) and bool(s.malformed_windows) for s in train),
                "observed_compile_labels": sum(s.compile_ok is not None for s in train),
                "syntax_proxy_is_not_compilation": True,
                "training_content_sha256": hashlib.sha256(json.dumps(training_content).encode()).hexdigest()}}
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_dir/"weights.pt")
        json_save(artifact_dir/"metadata.json", metadata)
    return export_mapping(events, metadata), metadata


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", default="Code-S", choices=["Code-S", "BePKT"])
    parser.add_argument("--mode", choices=["fit-transform", "transform"], default="fit-transform")
    parser.add_argument("--data-dir")
    parser.add_argument("--input-jsonl", help="Normalized submissions; score is [0,1], reason has five entries")
    parser.add_argument("--code-file")
    parser.add_argument("--main-file")
    parser.add_argument("--score-file")
    parser.add_argument("--reason-file")
    parser.add_argument("--split-file", help="JSON student -> split, or train/validation/test -> student ID lists; align with KT folds")
    parser.add_argument("--artifact-dir", help="Frozen vocabularies/encoders/centroids/graph directory")
    parser.add_argument("--out", help="Default: data/<dataset>/ast_kc_dual_mapping.json (does not replace old mapping)")
    parser.add_argument("--max-samples", "--max_samples", type=int, help="0=all; default BePKT=20000, Code-S=all")
    parser.add_argument("--language", choices=["java", "c", "cpp"], help="Override per-submission Language")
    parser.add_argument("--score-max", type=float, default=50.)
    parser.add_argument("--main-score-max", type=float, default=1., help="Scale of MainTable.Score, independently of CodeScore.csv")
    parser.add_argument("--compile-column", help="Explicit compiler-result column; numeric judge result codes are not guessed")
    parser.add_argument("--compile-success-values", nargs="*", default=[])
    parser.add_argument("--compile-failure-values", nargs="*", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output/artifact files")
    for name, field_info in Config.__dataclass_fields__.items():
        default = field_info.default
        option = "--" + name.replace("_", "-")
        aliases = [option, "--" + name] if "_" in name else [option]
        if isinstance(default, bool):
            parser.add_argument(*aliases, dest=name, action="store_true", default=default)
        else:
            parser.add_argument(*aliases, dest=name, type=type(default), default=default)
    args = parser.parse_args(argv)
    positive = ["score_max", "main_score_max", "max_tokens", "max_patterns", "batch_size", "epochs",
                "concept_dim", "malformed_dim", "concept_clusters", "malformed_clusters_per_family",
                "temperature_c", "temperature_m", "min_transition_support", "transition_stability", "learning_rate"]
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.assignment_threshold < 1:
        parser.error("assignment_threshold must be in [0,1)")
    if args.max_depth < 0 or args.context_window < 1 or args.graph_neighbors < 0 or args.reason_weight < 0 or (args.max_samples is not None and args.max_samples < 0):
        parser.error("Depth, neighbors, reason weight and sample limit cannot be negative")
    directory = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parents[2]/"data"/args.dataset
    args.out = Path(args.out) if args.out else directory/"ast_kc_dual_mapping.json"
    args.artifact_dir = Path(args.artifact_dir) if args.artifact_dir else args.out.parent/(args.out.stem + "_artifacts")
    if args.out.exists() and not args.overwrite:
        parser.error(f"Output exists: {args.out}; choose a new --out or explicitly --overwrite")
    if args.mode == "fit-transform" and args.artifact_dir.exists() and any(args.artifact_dir.iterdir()) and not args.overwrite:
        parser.error(f"Artifact directory is nonempty: {args.artifact_dir}; choose a new directory or --overwrite")
    if args.mode == "transform" and not (args.artifact_dir/"metadata.json").is_file():
        parser.error("transform requires an existing --artifact-dir containing metadata.json and weights.pt")
    return args


def main(argv=None):
    args = arguments(argv)
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    submissions, splits = load_submissions(args)
    print(f"Loaded {len(submissions)} events / {len({s.student_id for s in submissions})} students")
    if not args.split_file and args.mode == "fit-transform":
        print("Student hash split generated. Reuse exported student_splits for KT; do not independently re-split.")
    mapping, _ = run_pipeline(submissions, splits, config, args.artifact_dir, args.mode, args.device)
    json_save(args.out, mapping)
    print(f"Saved {mapping['concept_kcs']['num_kcs']} Conceptual / {mapping['malformed_kcs']['num_kcs']} Malformed KCs to {args.out}")
    print("KT must consume submission_kc_map[CodeStateID].kc_mask or typed branch weights; problem_kc_map is auxiliary only.")
    if mapping["graph"]["eligible_temporal_windows"] == 0:
        print("No eligible observed-compile temporal windows: MC edges remain zero (no inferred compiler verdicts).")


if __name__ == "__main__":
    main()
