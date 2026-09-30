"""Freeze local protocols and export train-only structural evidence; never fit models."""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import html
import importlib.metadata
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import ast_kc_discovery as kc

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/BePKT'
FINAL = BASE / 'dual_kc_taskA_final'
EVENTS = BASE / 'dual_kc_taskA_v3/event_master_with_code.jsonl'
OUT = ROOT / 'data/semantic_alignment_step12_v1'
SEED = 20260928
csv.field_size_limit(20_000_000)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    kc.json_save(path, value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_hash(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def split_map(raw):
    result = {}
    for label, ids in raw.items():
        for sid in ids:
            assert str(sid) not in result, 'Student overlap inside fold'
            result[str(sid)] = 'validation' if label == 'valid' else label
    assert set(result.values()) == {'train', 'validation', 'test'}
    return result


def load_fold(fold):
    artifact = FINAL / f'fold{fold}/post_order_fix/artifacts'
    args = kc.arguments(['BePKT', '--mode', 'transform', '--input-jsonl', str(EVENTS),
                         '--artifact-dir', str(artifact), '--max-samples', '0'])
    subs, _ = kc.load_submissions(args)
    meta = read(artifact / 'metadata.json')
    raw = read(FINAL / f'fold{fold}/student_split_raw.json')
    splits = split_map(raw)
    assert splits == meta['student_splits'], 'Saved split and model split disagree'
    train = [s for s in subs if splits.get(s.student_id) == 'train']
    content = sorted((s.student_id, s.problem_id, s.code_id, text_hash(s.code)) for s in train)
    fit_hash = text_hash(json.dumps(content))
    assert fit_hash == meta['fit_summary']['training_content_sha256'], 'Training content changed'
    assert len(train) == meta['fit_summary']['train_submissions']
    return artifact, meta, raw, splits, train, fit_hash


def cw_protocol():
    reports, identities = {}, {}
    # EventID namespaces can collide across datasets; compare source data within semester only.
    for tag in ('S', 'F'):
        directory = ROOT / f'data/Code-{tag}'
        students, problems, codes, dates, course = set(), set(), set(), [], Counter()
        core = Counter()
        with (directory / 'MainTable.csv').open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                students.add(row['SubjectID']); problems.add(row['ProblemID'])
                if row['CodeStateID']: codes.add(row['CodeStateID'])
                dates.append(row['ServerTimestamp']); course[row.get('CourseID', '')] += 1
                core[(row['ProblemID'], row['ServerTimestamp'], row['EventType'])] += 1
        with (directory / 'CodeStates.csv').open(encoding='utf-8-sig', newline='') as f:
            present = {row['CodeStateID'] for row in csv.DictReader(f) if row.get('Code', '').strip()}
        missing = codes - present
        affected = Counter()
        with (directory / 'MainTable.csv').open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row['CodeStateID'] in missing: affected[row['EventType']] += 1
        ids = sorted(students, key=lambda s: text_hash(f'{SEED}|CW-{tag}|{s}'))
        ntrain, nval = int(.6 * len(ids)), int(.2 * len(ids))
        split = {'train': ids[:ntrain], 'validation': ids[ntrain:ntrain+nval], 'test': ids[ntrain+nval:]}
        assert len(split_map(split)) == len(students)
        save(OUT / f'splits/cw_{tag.lower()}_split.json', split)
        identities[tag] = students
        reports[tag] = {'students': len(students), 'problems': len(problems), 'event_rows': len(dates),
                        'date_min': min(dates), 'date_max': max(dates), 'courses': dict(course),
                        'problem_ids': sorted(problems), 'code_state_ids': len(codes),
                        'code_states_missing_payload': len(codes - present),
                        'missing_code_state_ids': sorted(missing), 'missing_code_event_types': dict(affected),
                        'code_eligibility_policy': 'Exclude events with absent/empty code from code-based analysis; preserve raw event order and report attrition. No synthetic code.',
                        'split_counts': {k: len(v) for k, v in split.items()},
                        'source_hashes': {name: digest(directory / name) for name in ('MainTable.csv', 'CodeStates.csv')},
                        'semester_evidence': 'timestamp year/month; fall TermID not available' if tag == 'F' else 'pending TermID join'}
        if tag == 'S':
            original = ROOT / 'data/CodeWorkout/CodeWorkout1/MainTable.csv'
            terms, original_core = Counter(), Counter()
            with original.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    terms[row['TermID']] += 1
                    original_core[(row['ProblemID'], row['ServerTimestamp'], row['EventType'])] += 1
            assert core == original_core and set(terms) == {'spring-2019'}
            reports[tag]['semester_evidence'] = 'Problem/timestamp/event-type multiset matches local raw source with TermID=spring-2019; student/code IDs differ (anonymized), not identity-joined.'
            reports[tag]['raw_source_sha256'] = digest(original)
    save(OUT / 'protocol/codeworkout_audit.json', {'datasets': reports,
         'raw_subject_id_overlap': len(identities['S'] & identities['F']),
         'identity_limitation': 'Different IDs do not prove different real people; no cross-semester identity crosswalk available.',
         'directions': ['CW-S train -> CW-S validation -> entire CW-F external test',
                        'CW-F train -> CW-F validation -> entire CW-S external test'],
         'target_policy': 'Each direction has independent source-only artifacts; never tune on target semester.',
         'problem_policy': 'Retain semester-qualified IDs; equal raw problem IDs are not evidence of identical content.',
         'seed': SEED})


def freeze():
    reports = []
    for fold in range(5):
        artifact, meta, raw, splits, train, fit_hash = load_fold(fold)
        assert meta['graph']['fit_split'] == 'train'
        discovery = read(artifact.parent / 'discovery_audit.json')
        assert digest(EVENTS) == discovery['input']['sha256'], 'Source file differs from recorded discovery input'
        source_audit = read(artifact.parent / 'model_source_independent_audit.json')
        assert digest(artifact / 'weights.pt') == source_audit['formal_model_weights_sha256']
        save(OUT / f'splits/bepkt_fold{fold}.json', raw)
        problem_sets = defaultdict(set)
        with EVENTS.open(encoding='utf-8-sig') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    label = splits.get(str(event['student_id']))
                    if label is not None: problem_sets[label].add(str(event['problem_id']))
        reports.append({'fold': fold, 'split_counts': dict(Counter(splits.values())),
                        'problem_ids_by_split': {k: sorted(v) for k, v in problem_sets.items()},
                        'actual_train_students': len({s.student_id for s in train}), 'train_events': len(train),
                        'training_content_sha256': fit_hash, 'train_only_checks_pass': True,
                        'kc_counts': {b: meta['clusters'][b]['num_kcs'] for b in ('C', 'M')},
                        'source_artifacts': str(artifact.relative_to(ROOT)),
                        'source_hashes': {name: digest(artifact / name) for name in ('weights.pt', 'metadata.json')},
                        'split_sha256': digest(OUT / f'splits/bepkt_fold{fold}.json')})
        print(f'Protocol fold{fold}: verified {len(train)} train events', flush=True)
    cw_protocol()
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'pandas', 'tree-sitter', 'tree-sitter-c', 'tree-sitter-cpp', 'tree-sitter-java')}
    save(OUT / 'protocol/locked_protocol.json', {
        'schema': 'semantic-step12-v1', 'seed': SEED, 'folds': reports,
        'input_sha256': digest(EVENTS), 'problem_source_sha256': digest(BASE / 'problem.csv'),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_has_existing_changes': True,
        'implementation_hashes': {p.name: digest(p) for p in (Path(__file__), Path(kc.__file__))},
        'python': platform.python_version(), 'packages': versions,
        'prompt_version': None, 'prompt_status': 'No LLM generation in steps 1/2; freeze before step3.',
        'rules': {
            'namespace': 'dataset/fold/KC; no semantic label sharing across folds',
            'evidence': 'Only fold train students, including paired AC and AST statistics',
            'validation': 'Predeclared hyperparameter/model selection only; excluded from discovery and semantic grounding',
            'test': 'Final evaluation only; no taxonomy/prompt selection using test results',
            'task_A': 'Future AC only offline evidence/retrospective labels; never a prediction-time feature',
            'malformed': 'Compiler-confirmed failure patterns; no automatic misconception or logical-error claim',
            'members': 'Reconstruct original eligible truncated-token groups; first occurrence vector; assign to frozen family centers',
            'numerics': 'Preserve original full training batch composition when replaying frozen encoder; stop on any membership-statistic discrepancy',
            'selection': 'nearest center + 2 near + median + boundary; prioritize unique patterns/codes; disclose shortfalls',
            'pairing': 'Same train student/problem/language, strictly later raw problem order, OJ result=0; nearest and last AC',
            'ast_diff': 'Node/path multiset changes and parser error locations, not exact tree-edit alignment or causal repair proof',
            'legacy_split': 'Ignore input split; use copied per-fold student files'},
        'status': 'locked'})


def structural_stats(code, language):
    raw = code.encode('utf-8')
    root = kc.parser_for(language).parse(raw).root_node
    offset = 0
    if language == 'java' and root.has_error:
        prefix = b'class __KC_WRAPPER__ {\n'
        candidate = kc.parser_for(language).parse(prefix + raw + b'\n}').root_node
        error_count = lambda r: sum(n.type == 'ERROR' or n.is_missing for n in kc.walk(r))
        if error_count(candidate) < error_count(root): root, offset = candidate, len(prefix)
    nodes, paths, errors = Counter(), Counter(), []
    for n in kc.walk(root):
        if n.start_byte < offset or n.end_byte > offset + len(raw): continue
        if n.is_named:
            nodes[n.type] += 1
            path = list(reversed(kc.ancestors(n)[:2])) + [n.type]
            paths['/'.join(path)] += 1
        if n.type == 'ERROR' or n.is_missing:
            errors.append({'type': n.type, 'missing': n.is_missing,
                           'start_byte': n.start_byte-offset, 'end_byte': n.end_byte-offset})
    return {'nodes': dict(nodes), 'paths_depth3': dict(paths), 'parse_errors': errors,
            'java_wrapper_bytes': offset}


def pair_index(train, raw_events):
    groups = defaultdict(list)
    for sub in train:
        if raw_events[sub.event_id].get('oj_result') == 0:
            groups[(sub.student_id, sub.problem_id, sub.language)].append(sub)
    for group in groups.values(): group.sort(key=lambda s: (s.problem_attempt_order, s.event_id))
    return groups


def later_ac(sub, groups):
    return [s for s in groups.get((sub.student_id, sub.problem_id, sub.language), [])
            if s.problem_attempt_order > sub.problem_attempt_order]


def choose_members(members):
    ordered = sorted(members, key=lambda x: (x['distance'], x['pattern_hash']))
    median = float(np.median([m['distance'] for m in ordered]))
    ranked = [('center_representative', ordered), ('near_center_1', ordered), ('near_center_2', ordered),
              ('middle', sorted(ordered, key=lambda x: (abs(x['distance']-median), x['pattern_hash']))),
              ('boundary', list(reversed(ordered)))]
    selected, used_patterns, used_codes, used_students = [], set(), set(), set()
    for role, candidates in ranked:
        options = [m for m in candidates if m['pattern_hash'] not in used_patterns]
        if not options: break
        found = None
        for m in options:
            occurrences = sorted(m['occurrences'], key=lambda sp: (sp[0].student_id in used_students, sp[0].event_id, sp[1].start_byte))
            for s, p in occurrences:
                if text_hash(s.code) not in used_codes:
                    found = m, s, p; break
            if found: break
        if found is None: break
        m, s, p = found
        selected.append((role, m, s, p)); used_patterns.add(m['pattern_hash'])
        used_codes.add(text_hash(s.code)); used_students.add(s.student_id)
    return selected


def pattern_paths(pattern):
    """Read real parent-child structure from balanced, untruncated serialization."""
    if pattern.branch == 'M':
        fields = dict(t.split(':', 1) for t in pattern.tokens if ':' in t)
        return [fields.get('grandparent', 'ROOT') + '/' + fields.get('parent', 'ROOT') + '/' + pattern.root_type]
    stack, last, result = [], None, []
    for token in pattern.tokens:
        if token.startswith('node:'):
            last = token[5:]
            result.append('/'.join(stack[-2:] + [last]))
        elif token == '<CHILDREN>':
            assert last is not None
            stack.append(last)
        elif token == '</CHILDREN>':
            assert stack
            stack.pop()
    assert not stack
    return result


def export_fold(fold, device):
    artifact, meta, raw_split, splits, train, _ = load_fold(fold)
    config = kc.Config(**meta['config'])
    eligible = [s for s in train if kc.conceptual_prototype(s) or kc.malformed_prototype(s)]
    # Preserve original training batches: fused GRU kernels can differ slightly
    # with batch composition, which matters for nearly tied cluster boundaries.
    print(f'fold{fold}: parsing {len(train)} train submissions ({len(eligible)} prototype eligible)', flush=True)
    for i, s in enumerate(train):
        kc.parse_submission(s, config)
        if i and i % 2000 == 0: print(f'fold{fold}: parsed {i}', flush=True)
    model = kc.DualKCModel(meta['vocab_c'], meta['vocab_m'], config).to(device)
    model.load_state_dict(torch.load(artifact / 'weights.pt', map_location=device, weights_only=True))
    groups = {'C': {}, 'M': {}}
    for i, (s, zc, ac, zm, am) in enumerate(kc.representations(model, train, meta['vocab_c'], meta['vocab_m'], config, device)):
        for b, patterns, zs, weights, allowed in [('C', s.normal_subtrees, zc, ac, kc.conceptual_prototype(s)),
                                                ('M', s.malformed_windows, zm, am, kc.malformed_prototype(s))]:
            if not allowed: continue
            for p, z, weight in zip(patterns, zs, weights):
                if b == 'C' and p.damaged: continue
                key = (p.family, tuple(p.tokens[:config.max_tokens]))
                item = groups[b].setdefault(key, {'vector': z, 'weight': 0., 'occurrences': [],
                       'pattern_hash': text_hash(json.dumps(key)), 'family': p.family})
                item['weight'] += max(float(weight), 1e-8)
                item['occurrences'].append((s, p))
        if i and i % 2000 == 0: print(f'fold{fold}: encoded {i}', flush=True)
    del model
    with EVENTS.open(encoding='utf-8-sig') as f:
        raw_events = {e['event_id']: e for line in f if line.strip() for e in [json.loads(line)]
                      if splits.get(str(e['student_id'])) == 'train'}
    pairs = pair_index(train, raw_events)
    with (BASE / 'problem.csv').open(encoding='utf-8-sig', newline='') as f:
        problems = {row['id']: row for row in csv.DictReader(f)}
    root = OUT / f'evidence/bepkt/fold{fold}'
    root.mkdir(parents=True, exist_ok=True)
    summaries, reconstruction = [], []
    for b in ('C', 'M'):
        cluster = meta['clusters'][b]
        assigned = defaultdict(list)
        for item in groups[b].values():
            indices = [j for j, family in enumerate(cluster['families']) if family == item['family']]
            assert indices
            distances = np.linalg.norm(np.asarray(cluster['centers'])[indices] - item['vector'], axis=1)
            order = np.argsort(distances, kind='stable')
            j = indices[int(order[0])]
            item['distance'] = float(distances[order[0]])
            item['margin'] = float(distances[order[1]] - distances[order[0]]) if len(order) > 1 else None
            assigned[j].append(item)
        for j in range(cluster['num_kcs']):
            cid = f'{b}{j:03d}'; dest = root / cid; dest.mkdir(exist_ok=True)
            members = assigned[j]; expected = cluster['patterns'][str(j)]
            count = sum(len(m['occurrences']) for m in members)
            support = len({s.student_id for m in members for s, p in m['occurrences']})
            checks = {'num_members': count == expected['num_members'],
                      'unique_patterns': len(members) == expected['unique_patterns'],
                      'student_support': support == expected['student_support']}
            reconstruction.append({'kc_id': cid, 'checks': checks, 'actual': [count, len(members), support],
                                   'expected': [expected['num_members'], expected['unique_patterns'], expected['student_support']]})
            if not all(checks.values()):
                save(root / 'reconstruction_failure.json', {'cluster': reconstruction[-1],
                     'closest_boundary_patterns': [{'pattern_hash': m['pattern_hash'], 'distance': m['distance'], 'margin': m['margin'],
                       'occurrences': len(m['occurrences'])} for m in sorted(members,key=lambda x:x['margin'] if x['margin'] is not None else float('inf'))[:10]]})
            assert all(checks.values()), f'fold{fold}/{cid}: cannot reproduce original cluster {reconstruction[-1]}'
            with (dest / 'members_index.jsonl').open('w', encoding='utf-8') as f:
                for m in members:
                    for s, p in m['occurrences']:
                        f.write(json.dumps({'event_id': s.event_id, 'code_id': s.code_id, 'student_id': s.student_id,
                            'problem_id': s.problem_id, 'language': s.language, 'start_byte': p.start_byte,
                            'end_byte': p.end_byte, 'pattern_hash': m['pattern_hash'], 'distance': m['distance'],
                            'margin_to_second_center': m['margin'], 'split': 'train'}, ensure_ascii=False)+'\n')
            tokens, roots, paths = Counter(), Counter(), Counter()
            for m in members:
                for s, p in m['occurrences']:
                    roots[p.root_type] += 1
                    tokens.update(set(p.tokens[:config.max_tokens]))
                    paths.update(pattern_paths(p))
            assert dict(roots) == expected['type_distribution']
            selected = choose_members(members)
            # Add one genuine paired example when the structural selection has none.
            if b == 'M' and not any(later_ac(s, pairs) for _, _, s, _ in selected):
                candidates = [(m, s, p) for m in sorted(members, key=lambda x:x['distance'])
                              for s, p in m['occurrences'] if later_ac(s, pairs)]
                if candidates:
                    m, s, p = candidates[0]; selected.append(('paired_supplement', m, s, p))
            example_rows, paired_count = [], 0
            for n, (role, m, s, p) in enumerate(selected):
                assert splits[s.student_id] == 'train'
                assert 0 <= p.start_byte <= p.end_byte <= len(s.code.encode('utf-8'))
                example_dir = dest / f'examples/{n:02d}_{role}'; example_dir.mkdir(parents=True, exist_ok=True)
                extension = {'c': 'c', 'cpp': 'cpp', 'java': 'java'}[s.language]
                (example_dir / f'code.{extension}').write_text(s.code, encoding='utf-8', newline='')
                fragment = s.code.encode('utf-8')[p.start_byte:p.end_byte].decode('utf-8')
                (example_dir / f'fragment.{extension}').write_text(fragment, encoding='utf-8', newline='')
                problem = problems.get(s.problem_id)
                if problem:
                    fields = ('title', 'description', 'input_description', 'output_description', 'samples', 'hint')
                    save(example_dir / 'problem.json', {'problem_id': s.problem_id, **{k: problem[k] for k in fields}})
                    (example_dir / 'problem.txt').write_text('\n\n'.join(k+'\n'+html.unescape(re.sub('<[^>]+>', ' ', problem[k])) for k in fields), encoding='utf-8')
                info = {'role': role, 'event_id': s.event_id, 'code_id': s.code_id, 'student_id': s.student_id,
                        'problem_id': s.problem_id, 'language': s.language, 'split': 'train',
                        'code_sha256': text_hash(s.code), 'start_byte': p.start_byte, 'end_byte': p.end_byte,
                        'empty_missing_token_span': p.start_byte == p.end_byte,
                        'root_type': p.root_type, 'tokens': p.tokens, 'pattern_hash': m['pattern_hash'],
                        'distance': m['distance'], 'problem_found': problem is not None,
                        'path': str(example_dir.relative_to(OUT)), 'compile_ok': s.compile_ok,
                        'oj_result': raw_events[s.event_id].get('oj_result')}
                save(example_dir / 'source.json', info)
                ast_before = structural_stats(s.code, s.language)
                save(example_dir / 'ast.json', ast_before)
                if b == 'M':
                    correct = later_ac(s, pairs)
                    info['has_later_ac'] = bool(correct)
                    if correct: paired_count += 1
                    pair_rows = []
                    for kind, target in ([('nearest', correct[0]), ('final', correct[-1])] if correct else []):
                        assert splits[target.student_id] == 'train' and target.student_id == s.student_id
                        assert target.problem_id == s.problem_id and target.language == s.language
                        assert raw_events[target.event_id]['oj_result'] == 0
                        (example_dir / f'{kind}_ac.{extension}').write_text(target.code, encoding='utf-8', newline='')
                        after = structural_stats(target.code, target.language)
                        diff = {'kind': kind, 'wrong_event_id': s.event_id, 'ac_event_id': target.event_id,
                                'ac_code_sha256': text_hash(target.code), 'ac_split': 'train',
                                'wrong_attempt_order': s.problem_attempt_order, 'ac_attempt_order': target.problem_attempt_order,
                                'raw_same_problem_gap': target.problem_attempt_order-s.problem_attempt_order,
                                'text_unified_diff': list(difflib.unified_diff(s.code.splitlines(), target.code.splitlines(), fromfile='wrong', tofile=kind+'_ac', lineterm='')),
                                'before_errors': ast_before['parse_errors'], 'after_errors': after['parse_errors'],
                                'interpretation': 'Observed changes, not proof of minimal or causal repair'}
                        for key in ('nodes', 'paths_depth3'):
                            before_c, after_c = Counter(ast_before[key]), Counter(after[key])
                            diff[key+'_added'] = dict(after_c-before_c); diff[key+'_removed'] = dict(before_c-after_c)
                        pair_rows.append(diff)
                    save(example_dir / 'diff.json', {'status': 'paired' if correct else 'no_later_same_student_problem_language_AC', 'pairs': pair_rows})
                example_rows.append(info)
            paired_events = {s.event_id for m in members for s, p in m['occurrences'] if b == 'M' and later_ac(s, pairs)}
            unique_events = {s.event_id for m in members for s, p in m['occurrences']}
            # Structural tokens summarize full membership; whole AST stats are selected examples only.
            save(dest / 'ast_stats.json', {'scope': 'all eligible training pattern occurrences',
                 'root_types': dict(roots), 'structural_token_document_frequency': tokens.most_common(100),
                 'node_tokens': [(k, v) for k, v in tokens.most_common() if k.startswith('node:')],
                 'high_frequency_structural_paths': paths.most_common(50),
                 'high_frequency_patterns': [{'pattern_hash': m['pattern_hash'], 'occurrences': len(m['occurrences']),
                    'tokens': m['occurrences'][0][1].tokens[:config.max_tokens]}
                    for m in sorted(members, key=lambda x: -len(x['occurrences']))[:10]],
                 'representative_AST_stats': 'examples/*/ast.json; nodes and actual depth-3 ancestor paths'})
            row = {'fold': fold, 'kc_id': cid, 'namespace': f'BePKT/fold{fold}/{cid}',
                   'branch': 'Concept' if b == 'C' else 'Malformed', 'family': expected['family'],
                   'centroid': cluster['centers'][j], 'cluster_size_occurrences': count,
                   'unique_patterns': len(members), 'student_support': support,
                   'root_distribution': dict(roots), 'original_description': expected['description'],
                   'representative_kind': 'nearest_center_member (not pairwise-distance medoid)',
                   'membership_reconstruction_pass': True, 'examples': example_rows,
                   'representative_count': len(selected), 'minimum_four_distinct_patterns': len(selected) >= 4,
                   'pairable_events': len(paired_events) if b == 'M' else None,
                   'unique_events': len(unique_events), 'selected_paired_examples': paired_count if b == 'M' else None,
                   'status': 'ready' if len(selected) >= 4 and all(x['problem_found'] for x in example_rows) and (b == 'C' or paired_count) else 'needs_review'}
            save(dest / 'metadata.json', row)
            summaries.append({k: row[k] for k in ('fold', 'kc_id', 'branch', 'cluster_size_occurrences', 'unique_patterns',
                             'student_support', 'representative_count', 'pairable_events', 'unique_events', 'selected_paired_examples', 'status')})
    save(root / 'reconstruction_audit.json', {'all_checks_pass': True, 'clusters': reconstruction,
          'note': 'All membership counts, unique patterns, student support and root distributions match frozen metadata.'})
    save(root / 'summary.json', {'fold': fold, 'clusters': summaries,
         'total_kcs': len(summaries), 'ready_kcs': sum(s['status']=='ready' for s in summaries)})
    print(f'fold{fold}: exported {len(summaries)} packages', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['freeze', 'evidence', 'all'], default='all')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage in ('freeze', 'all'): freeze()
    if args.stage in ('evidence', 'all'):
        protocol = read(OUT / 'protocol/locked_protocol.json')
        assert protocol['input_sha256'] == digest(EVENTS)
        for fold in args.folds: export_fold(fold, args.device)


if __name__ == '__main__':
    main()
