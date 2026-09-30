"""Canonical BePKT + submission-level Dual-KC -> KT detailed CSVs.

No raw CSV joins, KC discovery, ID re-fitting or random student re-splitting.
Run prepare_bepkt_submissions.py and Dual-KC discovery before this exporter.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from prepare_bepkt_submissions import (
    DEFAULT_DATA_DIR, file_hash, load_json, normalize_splits, number,
    parse_reason, parse_skills, save_json, split_lists, timestamp,
)

SEQ_LEN = 200
OUT_COLUMNS = ["SubjectID", "ProblemID", "Skills", "KC_mask", "Reason", "Label", "ServerTimestamp", "TimeInterval",
               "CodeStateID", "RawSubjectID", "RawProblemID", "ScoreRaw", "ReasonMask", "CompileOK", "CompileSource",
               "EventID", "EventOrder", "SourceRowID", "Language", "ConceptKCs", "MalformedKCs",
               "ConceptKCWeights", "MalformedKCWeights", "KCAttributionSource", "Split"]


def load_canonical(clean_dir):
    clean_dir = Path(clean_dir)
    records = []
    with (clean_dir/"submissions_clean.jsonl").open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            required = {"code_id", "student_id", "problem_id", "timestamp", "score", "reason", "event_order"}
            if not required.issubset(rec):
                raise ValueError(f"Canonical line {line_number} missing {sorted(required-set(rec))}")
            for key in ["code_id", "student_id", "problem_id"]:
                if rec[key] is None or not str(rec[key]).strip():
                    raise ValueError(f"Invalid {key} at canonical line {line_number}")
                rec[key] = str(rec[key])
            records.append(rec)
    students = {str(k): v for k, v in load_json(clean_dir/"student_id_map.json").items()}
    problems = {str(k): v for k, v in load_json(clean_dir/"problem_id_map.json").items()}
    for name, mapping, used in [("student", students, {r["student_id"] for r in records}),
                               ("problem", problems, {r["problem_id"] for r in records})]:
        if set(mapping) != used or any(type(v) is not int or v < 1 for v in mapping.values()):
            raise ValueError(f"Invalid or incomplete {name} ID map.")
        if set(mapping.values()) != set(range(1,len(mapping)+1)):
            raise ValueError(f"{name} ID map must be a bijection to contiguous IDs starting at 1.")
    splits = normalize_splits(load_json(clean_dir/"student_split.json"), students)
    seen = set()
    for rec in records:
        key = (rec["student_id"], rec["problem_id"], rec["event_order"])
        if key in seen:
            raise ValueError("Duplicate event_order within a student/problem trajectory.")
        seen.add(key)
        if rec.get("split", splits[rec["student_id"]]) != splits[rec["student_id"]]:
            raise ValueError("Canonical record split conflicts with student_split.json.")
        if rec.get("student_id_kt",students[rec["student_id"]]) != students[rec["student_id"]]:
            raise ValueError("Canonical remapped student ID conflicts with ID map.")
        if rec.get("problem_id_kt",problems[rec["problem_id"]]) != problems[rec["problem_id"]]:
            raise ValueError("Canonical remapped problem ID conflicts with ID map.")
    return records, students, problems, splits


def branch_ids(value, prefix):
    values = value.keys() if isinstance(value, dict) else value
    if not isinstance(values, (list, tuple)) and not isinstance(value, dict):
        return []
    result = []
    for entry in values:
        text = str(entry)
        if text.startswith(prefix):
            text = text[len(prefix):]
        v = number(text)
        if v is None or not v.is_integer() or v < 0:
            raise ValueError(f"Invalid {prefix} branch KC ID: {entry}")
        result.append(int(v))
    return sorted(set(result))


class SubmissionKCLookup:
    def __init__(self, mapping, splits):
        if "submission_kc_map" not in mapping:
            raise ValueError("Dual mapping requires submission_kc_map; ProblemID mapping is not a substitute.")
        if "student_splits" not in mapping:
            raise ValueError("KC mapping lacks student_splits; cannot verify discovery/KT split equality.")
        if normalize_splits(mapping["student_splits"]) != splits:
            raise ValueError("KC discovery student split differs from canonical/KT split. Refit on the matching train students.")
        count = mapping.get("num_kcs")
        if type(count) is not int or count < 1:
            raise ValueError("Dual mapping requires positive integer num_kcs.")
        self.count = count
        self.mapping = mapping["submission_kc_map"]
        self.events = defaultdict(list)
        for event in mapping.get("submission_events", {}).values():
            self.events[str(event["code_id"])].append(event)
        self.concept_count = mapping.get("concept_kcs",{}).get("num_kcs")
        self.malformed_count = mapping.get("malformed_kcs",{}).get("num_kcs")
        if self.concept_count is not None and self.malformed_count is not None:
            if self.concept_count + self.malformed_count != self.count:
                raise ValueError("Branch KC counts do not sum to num_kcs.")

    def get(self, record):
        cid = record["code_id"]
        candidates = [e for e in self.events.get(cid, []) if str(e.get("student_id")) == record["student_id"]
                      and str(e.get("problem_id")) == record["problem_id"]
                      and timestamp(e.get("timestamp")) == timestamp(record.get("timestamp"))]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous submission event attribution for {cid}")
        state = candidates[0] if candidates else self.mapping.get(cid)
        source = "submission_event" if candidates else "code_state"
        if state is None:
            return None
        if not isinstance(state, dict) or not isinstance(state.get("kc_mask"), list):
            raise ValueError(f"Invalid submission KC record for {cid}")
        mask = state["kc_mask"]
        if any(type(k) is not int or k < 0 or k >= self.count for k in mask):
            raise ValueError(f"KC mask out of range for {cid}")
        c = branch_ids(state.get("concept", state.get("concept_kcs", [])), "C")
        m = branch_ids(state.get("malformed", state.get("malformed_kcs", [])), "M")
        if self.concept_count is not None and self.malformed_count is not None:
            if any(k >= self.concept_count for k in c) or any(k >= self.malformed_count for k in m):
                raise ValueError(f"Branch-local KC ID out of range for {cid}")
            expected = set(c) | {self.concept_count+k for k in m}
            if expected != set(mask):
                raise ValueError(f"Branch/global KC masks disagree for {cid}")
        c_weights = state.get("concept_kcs", state.get("concept") if isinstance(state.get("concept"),dict) else {})
        m_weights = state.get("malformed_kcs", state.get("malformed") if isinstance(state.get("malformed"),dict) else {})
        for weights in [c_weights, m_weights]:
            if not isinstance(weights, dict) or any(number(v) is None or not 0 <= float(v) <= 1 for v in weights.values()):
                raise ValueError(f"Invalid KC soft weights for {cid}")
        # Score and Reason ALWAYS come from canonical data, never from mapping labels.
        return {"mask": sorted(set(mask)), "concept": c, "malformed": m,
                "concept_weights": c_weights, "malformed_weights": m_weights, "source": source}


def export_records(records, students, problems, splits, mapping=None, kc_mode="dual", missing_kc="error",
                   reason_policy="require-complete", seq_len=SEQ_LEN, min_interactions=2):
    if seq_len < 0 or min_interactions < 1:
        raise ValueError("seq_len must be nonnegative and min_interactions positive.")
    if kc_mode not in {"dual","skills","empty"} or missing_kc not in {"error","skip","empty"} or reason_policy not in {"preserve","require-complete"}:
        raise ValueError("Invalid export policy.")
    if kc_mode == "dual" and mapping is None:
        raise FileNotFoundError("Dual KC mapping is required; run discovery first, or explicitly select --kc-mode skills/empty.")
    lookup = SubmissionKCLookup(mapping, splits) if kc_mode == "dual" else None
    groups, stats = defaultdict(list), Counter()
    missing_examples = []
    for rec in records:
        score = number(rec["score"])
        if score is None:
            stats["skipped_missing_score"] += 1
            continue
        if not 0 <= score <= 1:
            raise ValueError(f"Canonical score outside [0,1]: {rec['code_id']}")
        ts = timestamp(rec["timestamp"])
        if ts is None:
            stats["skipped_missing_timestamp"] += 1
            continue
        reason = parse_reason(rec["reason"])
        if reason_policy == "require-complete" and any(v is None for v in reason):
            stats["skipped_incomplete_reason"] += 1
            continue
        skills = parse_skills(rec.get("skills", []))
        if kc_mode == "dual":
            state = lookup.get(rec)
            if state is None:
                stats["missing_kc_attribution"] += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rec["code_id"])
                if missing_kc in {"skip","error"}:
                    continue
                state = {"mask":[],"concept":[],"malformed":[],"concept_weights":{},"malformed_weights":{},"source":"missing_explicit_empty"}
        else:
            if kc_mode == "skills" and not skills:
                stats["skipped_missing_skills"] += 1
                continue
            state = {"mask":skills if kc_mode == "skills" else [],"concept":[],"malformed":[],
                     "concept_weights":{},"malformed_weights":{},"source":"manual_skills" if kc_mode == "skills" else "explicit_empty"}
        raw_sid, raw_pid = rec["student_id"], rec["problem_id"]
        if raw_sid not in students or raw_sid not in splits or raw_pid not in problems:
            raise ValueError("Canonical interaction has no saved ID/split mapping.")
        row = {"SubjectID": students[raw_sid], "ProblemID": problems[raw_pid],
               "RawSubjectID": raw_sid, "RawProblemID": raw_pid, "CodeStateID": rec["code_id"],
               "Skills": repr(skills), "KC_mask": repr(state["mask"]),
               "Reason": repr(reason), "ReasonMask": repr([int(v is not None) for v in reason]),
               "Label": score, "ScoreRaw": rec.get("score_raw"), "ServerTimestamp": ts,
               "CompileOK": rec.get("compile_ok"), "CompileSource": rec.get("compile_source","unknown"),
               "EventID": rec.get("event_id", ""), "EventOrder": rec["event_order"],
               "SourceRowID": rec.get("source_row_id",0), "Language": rec.get("language","unknown"),
               "ConceptKCs": repr(state["concept"]), "MalformedKCs": repr(state["malformed"]),
               "ConceptKCWeights": json.dumps(state["concept_weights"],allow_nan=False),
               "MalformedKCWeights": json.dumps(state["malformed_weights"],allow_nan=False),
               "KCAttributionSource": state["source"], "Split": splits[raw_sid],
               "_time": pd.Timestamp(ts).value, "_kc_count":len(state["mask"])}
        groups[raw_sid].append(row)
    if missing_kc == "error" and stats["missing_kc_attribution"]:
        raise ValueError(f"Missing submission attribution for {stats['missing_kc_attribution']} eligible records; examples={missing_examples}. "
                         "Run frozen discovery transform on the full canonical dataset; never fall back by ProblemID.")
    result = []
    for raw_sid in sorted(groups):
        group = sorted(groups[raw_sid], key=lambda r: (r["_time"],r["SourceRowID"],r["EventOrder"]))
        if seq_len:
            stats["truncated_valid_records"] += max(0,len(group)-seq_len)
            group = group[:seq_len]  # AFTER filtering and sorting
        if len(group) < min_interactions:
            stats["skipped_short_students"] += 1
            stats["skipped_short_student_records"] += len(group)
            continue
        previous = None
        for row in group:
            row["TimeInterval"] = 0. if previous is None else round((row["_time"]-previous)/60e9, 4)
            previous = row["_time"]
            stats["rows_with_more_than_10_kcs"] += row["_kc_count"] > 10
            del row["_time"], row["_kc_count"]
            result.append(row)
    stats["canonical_records"] = len(records)
    stats["exported_records"] = len(result)
    stats["exported_students"] = len({r["RawSubjectID"] for r in result})
    return pd.DataFrame(result, columns=OUT_COLUMNS), dict(stats)


def export(clean_dir, mapping_path, out_dir, kc_mode="dual", missing_kc="error", reason_policy="require-complete",
           seq_len=SEQ_LEN, min_interactions=2, overwrite=False):
    clean_dir, out_dir = Path(clean_dir), Path(out_dir)
    outputs = ["q_a_train_detailed.csv","q_a_valid_detailed.csv","q_a_test_detailed.csv", "code_map.csv",
               "student_id_map.json","problem_id_map.json","student_split.json","export_report.json"]
    if not overwrite and any((out_dir/name).exists() for name in outputs):
        raise FileExistsError("KT output already exists; choose --out-dir or explicitly --overwrite.")
    records, students, problems, splits = load_canonical(clean_dir)
    mapping = load_json(mapping_path) if kc_mode == "dual" and mapping_path and Path(mapping_path).is_file() else None
    frame, stats = export_records(records, students, problems, splits, mapping, kc_mode,
                                  missing_kc, reason_policy, seq_len, min_interactions)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, name in [("train","train"),("validation","valid"),("test","test")]:
        subset = frame[frame["Split"] == split]
        subset.to_csv(out_dir/f"q_a_{name}_detailed.csv",index=False,quoting=csv.QUOTE_NONNUMERIC,encoding="utf-8")
        counts[split] = {"records":len(subset),"students":subset["SubjectID"].nunique()}
    code_map = frame[["SubjectID","ProblemID","CodeStateID","RawSubjectID","RawProblemID","EventID"]].copy()
    code_map.insert(1,"idx",code_map.groupby("SubjectID",sort=False).cumcount())
    code_map.to_csv(out_dir/"code_map.csv",index=False)
    save_json(out_dir/"student_id_map.json",students)
    save_json(out_dir/"problem_id_map.json",problems)
    save_json(out_dir/"student_split.json",split_lists(splits))
    stats.update({"schema_version":"bepkt-kt-2.0","kc_mode":kc_mode,"missing_kc_policy":missing_kc,
                  "reason_policy":reason_policy,"seq_len":seq_len,"min_interactions":min_interactions,
                  "split_counts":counts,"canonical_sha256":file_hash(clean_dir/"submissions_clean.jsonl"),
                  "kc_mapping_sha256":file_hash(mapping_path) if mapping is not None else None,
                  "num_kcs":mapping.get("num_kcs") if mapping is not None else None,
                  "max_problem_id":max(problems.values(),default=0),"max_student_id":max(students.values(),default=0),
                  "reason_mask_required":reason_policy == "preserve",
                  "code_asset_notice":"Rebuild/align code_map and code features before enabling existing --code_assets."})
    save_json(out_dir/"export_report.json",stats)
    print(json.dumps({"output_dir":str(out_dir),"kc_mode":kc_mode,"reason_policy":reason_policy,**counts},ensure_ascii=False))
    if reason_policy == "preserve":
        print("Unknown Reason values remain None. KT loader/loss must explicitly consume ReasonMask.")
    if stats["rows_with_more_than_10_kcs"]:
        print("Some rows contain >10 KCs; update the old KT loader's max_length=10 before training.")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument("--clean-dir",type=Path)
    parser.add_argument("--kc-mapping",type=Path)
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--kc-mode",choices=["dual","skills","empty"],default="dual")
    parser.add_argument("--missing-kc",choices=["error","skip","empty"],default="error")
    parser.add_argument("--reason-policy",choices=["require-complete","preserve"],default="require-complete")
    parser.add_argument("--seq-len",type=int,default=SEQ_LEN,help="0=all valid KT interactions; canonical data is never truncated")
    parser.add_argument("--min-interactions",type=int,default=2)
    parser.add_argument("--overwrite",action="store_true")
    args = parser.parse_args(argv)
    mapping_path = args.kc_mapping or args.data_dir/"kc"/"ast_kc_dual_mapping.json"
    export(args.clean_dir or args.data_dir/"clean",mapping_path,args.out_dir or args.data_dir/"MSDGKT",
           args.kc_mode,args.missing_kc,args.reason_policy,args.seq_len,args.min_interactions,args.overwrite)


if __name__ == "__main__":
    main()
