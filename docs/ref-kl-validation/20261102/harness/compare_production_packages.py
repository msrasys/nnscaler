"""Compare all emitted rank source and per-rank metadata for identical input IR."""
import argparse,hashlib,json,pickle,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def normalized(s):
    for name in ['nnscaler-transfer','nnscaler-combined','nnscaler-refine','nnscaler-base','nnscaler']:
        s=s.replace(str(ROOT/name),'<nnscaler>')
    # The existing generator emits custom-op imports by iterating a set.
    # These standalone replays did not fix PYTHONHASHSEED. Normalize only
    # that audited import-only block, preserving every other statement.
    marker = re.search(r"(?m)^runtime_version = [^\n]+\n", s)
    assert marker is not None
    end = s.index('class GenModel(', marker.end())
    block = [line.strip() for line in s[marker.end():end].splitlines() if line.strip()]
    assert all(re.fullmatch(r'import [A-Za-z_][A-Za-z0-9_.]*', line) for line in block), block
    return s[:marker.end()] + '\n' + '\n'.join(sorted(block)) + '\n\n' + s[end:]
left={f.relative_to(a.left):f for f in a.left.rglob('gencode*.py')}
right={f.relative_to(a.right):f for f in a.right.rglob('gencode*.py')}
assert len(left)==len(right)==3072 and left.keys()==right.keys()
changed=[];manifest={}
for rel,f in sorted(left.items()):
    lh=hashlib.sha256(normalized(f.read_text()).encode()).hexdigest()
    rh=hashlib.sha256(normalized(right[rel].read_text()).encode()).hexdigest()
    manifest[str(rel)]={'left':lh,'right':rh}
    if lh!=rh:changed.append(str(rel))
metadata=[]
for src in sorted(a.left.rglob('attr_meta.pkl')):
    rel=src.relative_to(a.left);dst=a.right/rel
    with src.open('rb') as f:lm=pickle.load(f)
    with dst.open('rb') as f:rm=pickle.load(f)
    assert lm['version']==rm['version']
    assert len(lm['rank_to_variant'])==len(rm['rank_to_variant'])==1536
    bad=[]
    for rank,(lv,rv) in enumerate(zip(lm['rank_to_variant'],rm['rank_to_variant'])):
        if lm['unique_payloads'][lv]!=rm['unique_payloads'][rv]:
            if pickle.loads(lm['unique_payloads'][lv])!=pickle.loads(rm['unique_payloads'][rv]):bad.append(rank)
    metadata.append({'module':str(rel.parent),'ranks':1536,'different_ranks':bad})
ir=[]
for src in a.left.rglob('graph.ckp'):
    dst=a.right/src.relative_to(a.left)
    ir.append({'module':str(src.parent.relative_to(a.left)),'identical_input_ir':hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest()})
report={'source_files':len(left),'different_sources':changed,'metadata':metadata,'ir':ir,'normalization':'Audited NNScaler checkout prefixes plus sorting only the import-only custom-op registry block emitted from a Python set; all remaining source, comments and identifiers unchanged','source_sha256':manifest}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'source_files':len(left),'different_source_count':len(changed),'examples':changed[:3],'metadata':metadata,'ir':ir,'normalization':report['normalization']},indent=2))
assert not changed and len(metadata)==2 and all(not x['different_ranks'] for x in metadata) and all(x['identical_input_ir'] for x in ir)
