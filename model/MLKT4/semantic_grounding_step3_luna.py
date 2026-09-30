"""Prepare/audit train-only inputs and resume isolated Codex semantic generations.

External judges are intentionally deferred. No mock outputs or fabricated metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import importlib
import importlib.metadata
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/semantic_alignment_step12_v1'
OUT = ROOT / 'data/semantic_alignment_step3_v2_luna_14kc'
VERSION = 'grounding-v2-luna'
STATUSES = ['supported', 'mixed_cluster', 'ambiguous', 'insufficient_evidence']
PILOT = ['C000', 'C007', 'C015', 'C024', 'C039', 'M000', 'M004', 'M008', 'M012', 'M016']


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def arr(item):
    return {'type': 'array', 'items': item}


def schemas():
    string = {'type':'string'}; nullable = {'type':['string','null']}; strings=arr(string)
    claim=obj({'claim':string,'evidence_ids':strings})
    profile=obj({
        'namespace':string,'kc_id':string,'branch':{'type':'string','enum':['Concept','Malformed']},
        'concept_type':{'type':'string','enum':['structural_concept','malformed_pattern']},
        'semantic_status':{'type':'string','enum':STATUSES},
        'semantic_name':nullable,'semantic_category':nullable,'definition':nullable,
        'required_knowledge':strings,'observable_structural_behavior':strings,'possible_errors':strings,
        'error_pattern_name':nullable,'related_programming_concept':nullable,'error_description':nullable,
        'likely_error_explanation':nullable,'misconception_hypothesis':nullable,
        'misconception_confidence':{'type':'string','enum':['none','low','medium']},
        'repair_evidence':arr(claim),'observed_structural_changes':arr(claim),'evidence_summary':arr(claim),
        'alternative_interpretations':strings,'limitations':strings,
        'coverage_scope':{'type':'string','enum':['primary_only','selected_representatives']},
        'cluster_coverage':{'type':'string','enum':['not_checked','high','partial','low','unknown']},
        'confidence':{'type':'number','minimum':0,'maximum':1}})
    coverage=obj({'revised_profile':profile,'coverage_checks':arr(obj({'example_id':string,
        'assessment':{'type':'string','enum':['supported','inconsistent','unclear','missing']},
        'explanation':string,'evidence_ids':strings})),
        'revision_level':{'type':'string','enum':['none','minor','major']},'revision_summary':string})
    score=obj({'score':{'type':['integer','null'],'minimum':1,'maximum':5},'not_applicable':{'type':'boolean'},
               'justification':string,'evidence_ids':strings})
    judge=obj({'namespace':string,'evidence_consistency':score,'semantic_specificity':score,
               'cluster_coverage':score,'pedagogical_plausibility':score,'fatal_issues':strings})
    return {'profile':profile,'coverage':coverage,'judge':judge}


def assert_safe_keys(value):
    forbidden={'original_description','centroid','student_id','event_id','code_id','path','timestamp',
               'Judge score','student_id_raw','student_id_kt'}
    if isinstance(value,dict):
        assert not (forbidden & set(value)), f'Forbidden input keys: {forbidden & set(value)}'
        for v in value.values(): assert_safe_keys(v)
    elif isinstance(value,list):
        for v in value: assert_safe_keys(v)


def checked_read(path, manifest, text=False):
    key = Path(path).relative_to(SOURCE).as_posix()
    # Step12 manifest uses Windows path separators.
    expected=manifest.get(key, manifest.get(key.replace('/', '\\')))
    assert expected and sha(path)==expected, f'Unsealed/changed source: {key}'
    return Path(path).read_bytes().decode('utf-8') if text else load(path)


_PARSERS = {}
_PARSER_LANGUAGES = ('c', 'cpp', 'java')
_LANGUAGE_CUES = {
    'cpp': ((r'\busing\s+namespace\b', 'using namespace'),
            (r'\bstd\s*::', 'std::'), (r'\btemplate\s*<', 'template<'),
            (r'\b(?:cout|cin)\s*(?:<<|>>)', 'C++ streams'),
            (r'#\s*include\s*[<"](?:iostream|vector|string|map)[>"]', 'C++ header')),
    'java': ((r'\b(?:public|private|protected)\s+class\b', 'Java class'),
             (r'\bimport\s+java\.', 'Java import'),
             (r'\bSystem\s*\.\s*out\b', 'System.out'),
             (r'\bpublic\s+static\s+void\s+main\b', 'Java main')),
}


def parser_for(language):
    if language not in _PARSERS:
        from tree_sitter import Language, Parser
        grammar=importlib.import_module('tree_sitter_'+language)
        _PARSERS[language]=Parser(Language(grammar.language()))
    return _PARSERS[language]


def parser_counts(code, language):
    raw=code.encode('utf-8')
    root=parser_for(language).parse(raw).root_node
    offset=0
    if language=='java' and root.has_error:
        prefix=b'class __KC_WRAPPER__ {\n'
        wrapped=parser_for(language).parse(prefix+raw+b'\n}').root_node
        def error_total(node):
            return sum(n.type=='ERROR' or n.is_missing for n in walk_nodes(node))
        if error_total(wrapped)<error_total(root):root,offset=wrapped,len(prefix)
    nodes=[n for n in walk_nodes(root) if n.start_byte>=offset and n.end_byte<=offset+len(raw)]
    return {'error_nodes':sum(n.type=='ERROR' for n in nodes),
            'missing_nodes':sum(n.is_missing for n in nodes),
            'java_wrapper_used':bool(offset)}


def walk_nodes(root):
    stack=[root]
    while stack:
        node=stack.pop()
        yield node
        stack.extend(reversed(node.children))


def language_consistency_audit(code, recorded_language):
    """Conservative parser comparison; never changes the source language label."""
    assert recorded_language in _PARSER_LANGUAGES
    counts={language:parser_counts(code,language) for language in _PARSER_LANGUAGES}
    errors={lang:row['error_nodes']+row['missing_nodes'] for lang,row in counts.items()}
    best=min(_PARSER_LANGUAGES,key=lambda lang:(errors[lang],lang!=recorded_language))
    code_for_cues=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                         ' ',code)
    cues={lang:[label for pattern,label in patterns if re.search(pattern,code_for_cues)]
          for lang,patterns in _LANGUAGE_CUES.items()}
    alternatives=[lang for lang in _PARSER_LANGUAGES if lang!=recorded_language and cues.get(lang)
                  and errors[lang]==0 and errors[recorded_language]>=1]
    mismatch=bool(alternatives)
    # C++ accepts much C syntax, so parser ties or small differences are inconclusive.
    if mismatch:status='suspected_mismatch'
    elif errors[recorded_language]==0 and len(code.strip())>=20:status='consistent'
    else:status='uncertain'
    return {'recorded_language':recorded_language,'status':status,
            'parser_counts':counts,'parser_error_counts':errors,'best_parser_language':best,
            'source_syntax_cues':{k:v for k,v in cues.items() if v},
            'interpretation':'Parser comparison is heuristic; it does not relabel the source or prove a student error.'}


def build_input(meta_path, manifest):
    m=checked_read(meta_path,manifest); fold=m['fold']; namespace=m['namespace']
    split=checked_read(SOURCE/f'splits/bepkt_fold{fold}.json',manifest)
    train=set(map(str,split['train']))
    stats=checked_read(meta_path.parent/'ast_stats.json',manifest)
    references={'S.roots':m['root_distribution'], 'S.support':{k:m[k] for k in
                ('cluster_size_occurrences','unique_patterns','student_support','unique_events','pairable_events')},
                'S.nodes':stats['node_tokens'][:40], 'S.paths':stats['high_frequency_structural_paths'][:30]}
    for i, pattern in enumerate(stats['high_frequency_patterns']): references[f'S.pattern.{i}']=pattern
    primary, challenge, supplemental, trace, language_audits=[],[],[],[],[]
    for i,e in enumerate(m['examples']):
        assert e['student_id'] in train and e['split']=='train'
        eid=f'E{i}'; directory=SOURCE/e['path']; ext={'c':'c','cpp':'cpp','java':'java'}[e['language']]
        code=checked_read(directory/f'code.{ext}',manifest,text=True)
        fragment=checked_read(directory/f'fragment.{ext}',manifest,text=True)
        assert code.encode()[e['start_byte']:e['end_byte']].decode()==fragment
        language_audit=language_consistency_audit(code,e['language'])
        language_audits.append({'namespace':namespace,'example_id':eid,'role':e['role'],**language_audit})
        refs={eid+'.context':{'recorded_language':e['language'],'role':e['role'],
                              'language_audit':language_audit},
              eid+'.code':code, eid+'.focus':{'start_byte':e['start_byte'],'end_byte':e['end_byte'],
                  'start_line':code.encode()[:e['start_byte']].count(b'\n')+1,'fragment':fragment,
                  'root_type':e['root_type'],'tokens':e['tokens']},
              eid+'.problem':checked_read(directory/'problem.json',manifest),
              eid+'.parser':checked_read(directory/'ast.json',manifest)}
        assert language_audit['parser_error_counts'][e['language']]==len(refs[eid+'.parser']['parse_errors']), \
            f'Recorded-parser count differs from sealed Step2 AST: {namespace}/{eid}'
        refs[eid+'.problem'].pop('problem_id',None)
        info={'example_id':eid,'role':e['role'],'language':e['language'],'evidence_ids':list(refs)}
        if m['branch']=='Malformed':
            assert e['compile_ok'] is False
            refs[eid+'.compiler_fact']={'compile_ok':False,'source':'OJ result, not parser inference'}
            diff=checked_read(directory/'diff.json',manifest)
            seen=set()
            for p in diff['pairs']:
                if p['ac_event_id'] in seen: continue
                seen.add(p['ac_event_id'])
                ac=checked_read(directory/f"{p['kind']}_ac.{ext}",manifest,text=True)
                refs[eid+'.'+p['kind']+'_repair']={
                    'accepted_code':ac,'OJ_AC':True,'pair_policy':'same training student/problem/language, strictly later',
                    'raw_same_problem_gap':p['raw_same_problem_gap'],
                    'ast_pair_confidence':'low' if p['after_errors'] else 'normal',
                    **{k:p[k] for k in ('text_unified_diff','before_errors','after_errors','nodes_added','nodes_removed','paths_depth3_added','paths_depth3_removed')}}
            info['repair_status']=diff['status']
        info['evidence_ids']=list(refs)
        references.update(refs)
        target=primary if e['role'] in ('center_representative','near_center_1','near_center_2') else (
            challenge if e['role'] in ('middle','boundary') else supplemental)
        target.append(info)
        trace.append({'example_id':eid,'source_path':e['path'],'student_id':e['student_id'],'event_id':e['event_id']})
    common={'namespace':namespace,'kc_id':m['kc_id'],'branch':m['branch'],'restricted_generation':m['status']=='needs_review',
            'source_readiness':m['status'],'support_summary':references['S.support']}
    pkeys=[k for k in references if k.startswith('S.')]+[r for e in primary for r in e['evidence_ids']]
    a={**common,'primary_evidence':primary,'references':{k:references[k] for k in pkeys}}
    b={**common,'primary_evidence':primary,'challenge_evidence':challenge,'supplemental_evidence':supplemental,'references':references}
    assert_safe_keys(a);assert_safe_keys(b)
    return a,b,trace,language_audits


def prepare():
    manifest=load(SOURCE/'file_manifest.json')
    verification=checked_read(SOURCE/'verification.json',manifest)
    assert verification['verified_folds']==list(range(5)) and verification['all_integrity_checks_pass']
    for name,value in schemas().items():dump(OUT/f'schema/{name}.json',value)
    index=[]; language_audits=[]
    for path in sorted((SOURCE/'evidence/bepkt').glob('fold*/[CM]*/metadata.json')):
        a,b,trace,audits=build_input(path,manifest)
        language_audits.extend(audits)
        dest=OUT/'inputs'/a['namespace'].replace('BePKT/','')
        dump(dest/'primary.json',a);dump(dest/'coverage.json',b)
        dump(dest/'local_trace_not_for_model.json',trace)
        fold=int(a['namespace'].split('/')[1][4:])
        index.append({'namespace':a['namespace'],'fold':fold,'kc_id':a['kc_id'],'branch':a['branch'],
                      'primary_sha256':sha(dest/'primary.json'),'coverage_sha256':sha(dest/'coverage.json'),
                      'primary_characters':len(json.dumps(a,ensure_ascii=False)),'coverage_characters':len(json.dumps(b,ensure_ascii=False)),
                      'primary_examples':len(a['primary_evidence']),'challenge_examples':len(b['challenge_evidence']),
                      'restricted_generation':a['restricted_generation']})
    assert len(index)==300
    dump(OUT/'input_index.json',index)
    dump(OUT/'language_audit.json',{'version':VERSION,'parser_languages':list(_PARSER_LANGUAGES),
         'examples':language_audits})
    dump(OUT/'pilot_language_audit.json',{'version':VERSION,'examples':[x for x in language_audits
         if x['namespace'].split('/')[-2]=='fold0' and x['namespace'].split('/')[-1] in PILOT]})
    dump(OUT/'pilot_selection.json',{'fold':0,'kc_ids':PILOT,
         'rule':'Predeclared spread of Concept cluster IDs and one Malformed ID per structural family; no test results or generated semantics used.'})
    dump(OUT/'protocol.json',{'version':VERSION,'prepared_at':now(),'requested_model':'gpt-6-luna',
        'provider':'OpenAI via authenticated Codex CLI','immutable_model_snapshot':None,
        'model_snapshot_limitation':'CLI selects an alias; an immutable backend version has not been exposed.',
        'temperature':None,'top_p':None,'seed':None,'sampling_note':'Not set/exposed by this CLI path; do not invent values.',
        'reasoning_effort':'medium','runs_per_KC':3,'candidate_for_coverage':'run1 (predeclared; no best-of-three selection)',
        'source_manifest_sha256':sha(SOURCE/'file_manifest.json'),
        'language_audit_sha256':sha(OUT/'language_audit.json'),
        'parser_versions':{name:importlib.metadata.version(name) for name in
            ('tree-sitter','tree-sitter-c','tree-sitter-cpp','tree-sitter-java')},
        'prompt_hashes':{p.name:sha(p) for p in sorted((OUT/'prompt').glob('*.txt'))},
        'schema_hashes':{p.name:sha(p) for p in sorted((OUT/'schema').glob('*.json'))},
        'script_sha256':sha(__file__),'prompt_stage':'candidate_v2_pending_isolated_pilot_validation',
        'formal_judges':'deferred_by_user; no same-model surrogate scores',
        'metrics_policy':{'EGR':'resolvable reference IDs / all cited references; not truth entailment',
                          'SVS':None,'LLM_acceptance_rate':None,'semantic_stability':'pending three isolated generations and fixed semantic encoder'},
        'cross_fold_policy':'No profiles, candidate labels or generation responses shared across folds.',
        'pilot_overlap_caveat':'Fold0 training students may be held-out in other folds. Prompt pilot uses no test metrics; strict nested-CV claims would require an external or disjoint development set.',
        'completion_policy':'Generated profile and coverage are provisional pending external judges; unavailable outputs/metrics remain absent/null.'})
    print('Prepared 300 v2 KC inputs, language audit and 10 fold0 pilot IDs.',flush=True)


def validate_profile(profile,evidence,stage):
    jsonschema.validate(profile,schemas()['profile'])
    assert profile['namespace']==evidence['namespace'] and profile['kc_id']==evidence['kc_id'] and profile['branch']==evidence['branch']
    assert profile['coverage_scope']==('primary_only' if stage=='A' else 'selected_representatives')
    assert (profile['cluster_coverage']=='not_checked')==(stage=='A')
    refs=evidence['references']; total=0
    for field in ('evidence_summary','repair_evidence','observed_structural_changes'):
        for claim in profile[field]:
            assert claim['evidence_ids'], 'Unreferenced claim'
            assert set(claim['evidence_ids'])<=set(refs), 'Unknown evidence reference'
            total+=len(claim['evidence_ids'])
    if profile['semantic_status']=='supported': assert profile['evidence_summary'], 'Supported without evidence'
    if evidence['branch']=='Concept':
        assert profile['concept_type']=='structural_concept'
        assert all(profile[k] is None for k in ('error_pattern_name','related_programming_concept','error_description','likely_error_explanation','misconception_hypothesis'))
        assert not profile['repair_evidence'] and not profile['observed_structural_changes']
    else:
        assert profile['concept_type']=='malformed_pattern'
        assert all(profile[k] is None for k in ('semantic_name','semantic_category','definition'))
        assert not profile['required_knowledge'] and not profile['observable_structural_behavior'] and not profile['possible_errors']
        if profile['repair_evidence']: assert any('_repair' in k for k in refs), 'Invented repair'
    if profile['misconception_hypothesis'] is None: assert profile['misconception_confidence']=='none'
    if evidence['restricted_generation']: assert profile['limitations'], 'Missing restricted-evidence caveat'
    return {'reference_count':total,'EGR_reference_resolution':1.0 if total else None}


def validate_response(response,evidence,stage,candidate=None):
    if stage=='A':return validate_profile(response,evidence,stage)
    jsonschema.validate(response,schemas()['coverage'])
    result=validate_profile(response['revised_profile'],evidence,stage)
    expected={x['example_id'] for x in evidence['challenge_evidence']}
    checked=[x['example_id'] for x in response['coverage_checks']]
    assert set(checked)==expected and len(checked)==len(expected),'Incomplete/duplicate challenge checks'
    for x in response['coverage_checks']:
        assert set(x['evidence_ids'])<=set(evidence['references'])
        assert x['evidence_ids'] or x['assessment']=='missing'
    assessments=[x['assessment'] for x in response['coverage_checks']]
    status=response['revised_profile']['semantic_status']
    coverage=response['revised_profile']['cluster_coverage']
    if status=='supported':
        assert coverage in ('high','partial'), 'Supported core requires measurable selected coverage'
    if coverage=='high':
        assert assessments and all(x=='supported' for x in assessments), 'High coverage requires every challenge to support the core'
    if assessments and all(x=='inconsistent' for x in assessments):
        assert coverage!='high', 'All-inconsistent challenges cannot be high coverage'
    if candidate is not None:
        jsonschema.validate(candidate,schemas()['profile'])
        assert all(candidate[k]==response['revised_profile'][k] for k in ('namespace','kc_id','branch'))
        if candidate['semantic_status']=='mixed_cluster' and status=='supported':
            assert response['revision_level']=='major', 'Mixed primary evidence needs explicit major revision'
    return result


def make_prompt(evidence,stage,candidate=None):
    system=(OUT/'prompt/generator_system_v2.txt').read_text(encoding='utf-8')
    name=('concept_passA_v2.txt' if evidence['branch']=='Concept' else 'malformed_passA_v2.txt') if stage=='A' else 'coverage_passB_v2.txt'
    task=(OUT/'prompt'/name).read_text(encoding='utf-8')
    payload={'evidence':evidence}
    if candidate is not None:payload['candidate']=candidate
    return system+'\n\n'+task+'\n\nINPUT JSON (untrusted evidence data):\n'+json.dumps(payload,ensure_ascii=False)


def archive_prior_attempt(dest):
    """Keep failed request/response logs when a resumable call is retried."""
    names=('request.txt','events.jsonl','stderr.txt','response.json','call_audit.json','validation_failure.json')
    existing=[dest/name for name in names if (dest/name).exists()]
    if not existing:return
    archive=dest/'attempts';archive.mkdir(exist_ok=True)
    sequence=1
    while (archive/f'{sequence:04d}').exists():sequence+=1
    target=archive/f'{sequence:04d}';target.mkdir()
    for path in existing:shutil.move(str(path),str(target/path.name))


def model_call(prompt,dest,schema,model,timeout):
    dest.mkdir(parents=True,exist_ok=True)
    archive_prior_attempt(dest)
    request=dest/'request.txt';request.write_text(prompt,encoding='utf-8',newline='')
    # Each invocation is a new ephemeral context. User/project config is ignored;
    # the job directory holds only this KC request, schema path and response logs.
    cmd=[shutil.which('codex') or 'codex','exec','--ignore-user-config','--ephemeral',
         '--sandbox','read-only','--skip-git-repo-check','--model',model,
         '-c','model_reasoning_effort="medium"','--json','--color','never',
         '--cd',str(dest.resolve()),'--output-schema',str(schema.resolve()),
         '--output-last-message',str((dest/'response.json').resolve()),'-']
    started=now()
    try:
        completed=subprocess.run(cmd,input=prompt.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        stdout,stderr,exitcode=completed.stdout,completed.stderr,completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout,stderr,exitcode=error.stdout or b'',error.stderr or b'',-1
    (dest/'events.jsonl').write_bytes(stdout);(dest/'stderr.txt').write_bytes(stderr)
    failure=None
    content=(stdout+stderr).decode('utf-8',errors='replace')
    if 'usage limit' in content.lower():failure='provider_usage_limit'
    elif exitcode!=0:failure='process_or_transport_failure'
    events=[]
    for line in stdout.decode('utf-8',errors='replace').splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    for event in events:
        item=event.get('item',{})
        if item.get('type') in ('command_execution','file_change','mcp_tool_call','web_search'):
            failure='unexpected_tool_use'
    record={'status':'failed' if failure else 'returned','failure':failure,'model':model,'provider':'OpenAI/Codex',
            'model_version':None,'temperature':None,'top_p':None,'seed':None,'started_at':started,'finished_at':now(),
            'prompt_version':VERSION,'request_sha256':sha(request),'events_sha256':sha(dest/'events.jsonl'),
            'response_sha256':sha(dest/'response.json') if (dest/'response.json').exists() else None,
            'usage':[x.get('usage') for x in events if x.get('type')=='turn.completed'], 'exit_code':exitcode}
    dump(dest/'call_audit.json',record)
    if failure:raise RuntimeError(f'{failure}; stopping without fabricated output. See {dest}/call_audit.json')
    return load(dest/'response.json')


def run(fold,ids,timeout):
    protocol=load(OUT/'protocol.json')
    if fold!=0:
        gate=load(OUT/'fold0_gate.json')
        assert gate['passed'] and gate['protocol_sha256']==sha(OUT/'protocol.json'),'Fold0 gate not passed'
    if not ids:
        frozen=load(OUT/'prompt_frozen.json')
        assert frozen['protocol_sha256']==sha(OUT/'protocol.json'),'Prompt not frozen after isolated pilot'
    for name,value in protocol['prompt_hashes'].items():assert sha(OUT/'prompt'/name)==value,'Prompt changed; version and prepare required'
    for name,value in protocol['schema_hashes'].items():assert sha(OUT/'schema'/name)==value,'Schema changed'
    assert sha(__file__)==protocol['script_sha256'],'Runner changed; prepare required'
    assert sha(OUT/'language_audit.json')==protocol['language_audit_sha256'],'Language audit changed; prepare required'
    selected=[x for x in load(OUT/'input_index.json') if x['fold']==fold and (not ids or x['kc_id'] in ids)]
    assert selected
    for entry in selected:
        cid=entry['kc_id'];inp=OUT/f'inputs/fold{fold}/{cid}';dest=OUT/f'bepkt/fold{fold}/{cid}'
        a,b=load(inp/'primary.json'),load(inp/'coverage.json')
        assert sha(inp/'primary.json')==entry['primary_sha256'] and sha(inp/'coverage.json')==entry['coverage_sha256']
        for n in (1,2,3):
            target=dest/f'run{n}_raw.json'
            if target.exists():
                validate_profile(load(target),a,'A')
                previous=load(dest/f'calls/passA_run{n}/call_audit.json')
                assert previous['status']=='validated' and previous['parsed_output_sha256']==sha(target)
                assert previous['input_evidence_sha256']==entry['primary_sha256']
                assert previous['request_sha256']==hashlib.sha256(make_prompt(a,'A').encode()).hexdigest()
                continue
            job=dest/f'calls/passA_run{n}'
            response=model_call(make_prompt(a,'A'),job,OUT/'schema/profile.json',protocol['requested_model'],timeout)
            try:check=validate_response(response,a,'A')
            except Exception as error:
                dump(job/'validation_failure.json',{'error':str(error),'at':now()});raise
            dump(target,response)
            audit=load(job/'call_audit.json');audit.update({'namespace':a['namespace'],'fold':fold,'stage':'A','run':n,
                   'input_evidence_sha256':entry['primary_sha256'],'parsed_output_sha256':sha(target),'checks':check,'status':'validated'})
            dump(job/'call_audit.json',audit)
            print(f'{a["namespace"]} run{n} validated',flush=True)
        if not (dest/'coverage.json').exists():
            job=dest/'calls/passB'
            response=model_call(make_prompt(b,'B',load(dest/'run1_raw.json')),job,OUT/'schema/coverage.json',protocol['requested_model'],timeout)
            check=validate_response(response,b,'B',load(dest/'run1_raw.json'));dump(dest/'coverage.json',response)
            audit=load(job/'call_audit.json');audit.update({'namespace':a['namespace'],'fold':fold,'stage':'B',
                   'input_evidence_sha256':entry['coverage_sha256'],'parsed_output_sha256':sha(dest/'coverage.json'),'checks':check,'status':'validated'})
            dump(job/'call_audit.json',audit)
        response=load(dest/'coverage.json');validate_response(response,b,'B',load(dest/'run1_raw.json'))
        previous=load(dest/'calls/passB/call_audit.json')
        assert previous['status']=='validated' and previous['parsed_output_sha256']==sha(dest/'coverage.json')
        assert previous['input_evidence_sha256']==entry['coverage_sha256']
        assert previous['request_sha256']==hashlib.sha256(make_prompt(b,'B',load(dest/'run1_raw.json')).encode()).hexdigest()
        final={'profile':response['revised_profile'],'coverage_revision_level':response['revision_level'],
               'coverage_revision_summary':response['revision_summary'],'judge_status':'pending_external_models',
               'formal_semantic_acceptance':None,'SVS':None,'freeze_status':'provisional_pending_judges'}
        dump(dest/'final_profile.json',final)
        blinded=copy.deepcopy(final['profile']);blinded.pop('confidence')
        dump(OUT/f'judge_inputs/fold{fold}/{cid}.json',{'candidate':blinded,'evidence':b})
    summarize()


def summarize():
    raw,final,audits,archived_attempts=[],[],[],[]
    for p in sorted((OUT/'bepkt').glob('fold*/*/run*_raw.json')):
        raw.append({'run_file':str(p.relative_to(OUT)),'profile':load(p)})
    for p in sorted((OUT/'bepkt').glob('fold*/*/final_profile.json')):final.append(load(p))
    for p in sorted((OUT/'bepkt').glob('fold*/*/calls/*/call_audit.json')):audits.append(load(p))
    for p in sorted((OUT/'bepkt').glob('fold*/*/calls/*/attempts/*/call_audit.json')):
        archived_attempts.append({'audit_path':str(p.relative_to(OUT)),'audit':load(p)})
    for name,rows in [('semantic_profiles_raw.jsonl',raw),('semantic_profiles_final.jsonl',final)]:
        (OUT/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    revisions=[]
    for p in sorted((OUT/'bepkt').glob('fold*/*/coverage.json')):
        result=load(p)
        candidate=load(p.parent/'run1_raw.json')
        revisions.append({'namespace':candidate['namespace'],'branch':candidate['branch'],
            'passA_status':candidate['semantic_status'],
            'passB_status':result['revised_profile']['semantic_status'],
            'cluster_coverage':result['revised_profile']['cluster_coverage'],
            'revision_level':result['revision_level'],'revision_summary':result['revision_summary'],
            'coverage_assessments':{x['example_id']:x['assessment'] for x in result['coverage_checks']},
            'semantic_fields_changed':[key for key in ('semantic_name','semantic_category','definition',
                'error_pattern_name','related_programming_concept','error_description','likely_error_explanation')
                if candidate[key]!=result['revised_profile'][key]]})
    def distribution(rows,key):
        return {branch:dict(Counter(x[key] for x in rows if x['branch']==branch))
                for branch in ('Concept','Malformed')}
    audit_data=load(OUT/'language_audit.json') if (OUT/'language_audit.json').exists() else {'examples':[]}
    lang_counts=Counter(x['status'] for x in audit_data['examples'])
    dump(OUT/'coverage_revision_summary.json',{'version':VERSION,'completed_passB':len(revisions),
        'passA_status_distribution':distribution(revisions,'passA_status'),
        'passB_status_distribution':distribution(revisions,'passB_status'),
        'coverage_distribution':distribution(revisions,'cluster_coverage'),
        'revision_level_distribution':distribution(revisions,'revision_level'),
        'records':revisions})
    pilot=[]
    for cid in PILOT:
        dest=OUT/'bepkt/fold0'/cid; namespace=f'BePKT/fold0/{cid}'
        runs=[load(dest/f'run{n}_raw.json') if (dest/f'run{n}_raw.json').exists() else None for n in (1,2,3)]
        coverage=load(dest/'coverage.json') if (dest/'coverage.json').exists() else None
        examples=[x for x in audit_data['examples'] if x['namespace']==namespace]
        pilot.append({'namespace':namespace,'branch':'Concept' if cid.startswith('C') else 'Malformed',
            'passA_statuses':[x['semantic_status'] if x else None for x in runs],
            'all_three_passA_complete':all(x is not None for x in runs),
            'passA_status_agreement':len({x['semantic_status'] for x in runs})==1 if all(runs) else None,
            'passB_status':coverage['revised_profile']['semantic_status'] if coverage else None,
            'cluster_coverage':coverage['revised_profile']['cluster_coverage'] if coverage else None,
            'revision_level':coverage['revision_level'] if coverage else None,
            'language_audit_counts':dict(Counter(x['status'] for x in examples)),
            'misconception_hypothesis_in_run1':bool(runs[0]['misconception_hypothesis']) if runs[0] else None})
    dump(OUT/'pilot_diagnostics.json',{'version':VERSION,'complete_KCs':sum(x['passB_status'] is not None for x in pilot),
        'passA_status_agreement_is_not_semantic_correctness':True,'records':pilot})
    dump(OUT/'generation_audit.json',{'generated_at':now(),'input_KCs':300,'validated_independent_passA':len(raw),
        'provisional_final_profiles':len(final),'formal_judges_run':0,'formal_judges_status':'deferred_by_user',
        'SVS':None,'LLM_acceptance_rate':None,'semantic_stability':None,'semantic_embeddings_status':'pending_fixed_encoder_and_profiles',
        'language_audit_status_distribution':dict(lang_counts),
        'calls':audits,'archived_retry_attempts':archived_attempts,
        'same_conversation_pilot_excluded':True,
        'last_observed_execution_blocker':load(OUT/'execution_blocker.json') if (OUT/'execution_blocker.json').exists() else None,
        'completion':'incomplete' if len(final)<300 else 'generation_complete_external_validation_pending'})


def freeze_prompt():
    for cid in PILOT:
        inp=OUT/f'inputs/fold0/{cid}';dest=OUT/f'bepkt/fold0/{cid}'
        for n in (1,2,3):
            validate_profile(load(dest/f'run{n}_raw.json'),load(inp/'primary.json'),'A')
            audit=load(dest/f'calls/passA_run{n}/call_audit.json')
            assert audit['status']=='validated' and audit['parsed_output_sha256']==sha(dest/f'run{n}_raw.json')
        validate_response(load(dest/'coverage.json'),load(inp/'coverage.json'),'B',load(dest/'run1_raw.json'))
    dump(OUT/'prompt_frozen.json',{'frozen_at':now(),'protocol_sha256':sha(OUT/'protocol.json'),
        'pilot_KCs':PILOT,'note':'Schema/reference validation passed; not external semantic acceptance.'})


def embed(encoder_dir):
    """A real fixed local text encoder is required; no TF-IDF masquerading as semantics."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    encoder_dir=Path(encoder_dir).resolve()
    assert encoder_dir.is_dir(),'Supply a local downloaded encoder directory'
    hashes={str(p.relative_to(encoder_dir)):sha(p) for p in sorted(encoder_dir.rglob('*')) if p.is_file()}
    model=SentenceTransformer(str(encoder_dir),local_files_only=True)
    profiles=[load(p) for p in sorted((OUT/'bepkt').glob('fold*/*/final_profile.json'))]
    assert profiles,'No validated profiles'
    def text(p):
        if p['branch']=='Concept':
            fields=[p['semantic_name'],p['definition'],*p['required_knowledge'],*p['observable_structural_behavior']]
        else:
            fields=[p['error_pattern_name'],p['related_programming_concept'],p['error_description'],
                    p['likely_error_explanation']]
        return '\n'.join(str(v) for v in fields if v)
    texts=[text(x['profile']) for x in profiles]
    vectors=model.encode(texts,normalize_embeddings=True,convert_to_numpy=True)
    np.save(OUT/'semantic_embeddings.npy',vectors)
    stability=[]
    for wrapped in profiles:
        p=wrapped['profile'];rel=p['namespace'].replace('BePKT/','');dest=OUT/'bepkt'/rel
        runs=[load(dest/f'run{n}_raw.json') for n in (1,2,3)]
        strings=[(x['semantic_name'] or x['error_pattern_name'] or '')+'\n'+(x['definition'] or x['error_description'] or '') for x in runs]
        if any(not s.strip() for s in strings): score=None
        else:
            emb=model.encode(strings,normalize_embeddings=True,convert_to_numpy=True)
            score=float((emb[0]@emb[1]+emb[0]@emb[2]+emb[1]@emb[2])/3)
        stability.append({'namespace':p['namespace'],'pairwise_cosine':score,
            'status_agreement':len({x['semantic_status'] for x in runs})==1,'statuses':[x['semantic_status'] for x in runs]})
    dump(OUT/'semantic_stability.json',stability)
    dump(OUT/'embedding_manifest.json',{'encoder_files_sha256':hashes,'created_at':now(),
        'namespace_order':[x['profile']['namespace'] for x in profiles],
        'valid_semantic_mask':[x['profile']['semantic_status']=='supported' for x in profiles],
        'texts':texts,'shape':list(vectors.shape),'embeddings_sha256':sha(OUT/'semantic_embeddings.npy'),
        'external_judges_pending':True,
        'exclude_fields':['confidence','semantic_status','judge_scores','misconception_hypothesis',
                          'misconception_confidence','cluster_coverage','revision_level']})
    # Predeclared engineering gate, not evidence of scientific success.
    fold0=[s for s in stability if '/fold0/' in s['namespace']]
    cosine=[s['pairwise_cosine'] for s in fold0 if s['pairwise_cosine'] is not None]
    fold0_concepts=[load(p/'run1_raw.json') for p in sorted((OUT/'bepkt/fold0').glob('C*'))
                    if (p/'run1_raw.json').exists()]
    primary_supported=sum(p['semantic_status']=='supported' for p in fold0_concepts)
    passed=(len(fold0)==60 and len(cosine)>=48 and float(np.mean(cosine))>=.80
            and sum(s['status_agreement'] for s in fold0)>=48 and primary_supported>=21)
    dump(OUT/'fold0_gate.json',{'passed':bool(passed),'protocol_sha256':sha(OUT/'protocol.json'),
         'thresholds':{'complete_profiles':60,'minimum_nonempty_stability':48,'mean_cosine':.80,
                       'status_agreement_count':48,'concept_primary_supported_minimum':21},
         'actual':{'profiles':len(fold0),'nonempty':len(cosine),'mean_cosine':float(np.mean(cosine)) if cosine else None,
                   'status_agreement':sum(s['status_agreement'] for s in fold0),
                   'concept_primary_supported':primary_supported},
         'interpretation':'Operational continuation gate; no claim of human/external semantic validity.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','summarize','freeze-prompt','embed'])
    p.add_argument('--fold',type=int,default=0,choices=range(5));p.add_argument('--ids',nargs='*')
    p.add_argument('--pilot',action='store_true');p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--encoder-dir')
    a=p.parse_args()
    if a.stage=='prepare':prepare();summarize()
    elif a.stage=='run':
        try:run(a.fold,PILOT if a.pilot else a.ids,a.timeout)
        finally:summarize()
    elif a.stage=='freeze-prompt':freeze_prompt()
    elif a.stage=='embed':
        if not a.encoder_dir:p.error('--encoder-dir required')
        embed(a.encoder_dir)
    else:summarize()


if __name__=='__main__':main()
