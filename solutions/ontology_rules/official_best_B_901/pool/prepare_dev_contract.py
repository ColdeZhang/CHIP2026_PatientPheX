"""Bind existing input-only dev artifacts for one no-model pool reconstruction."""
import argparse,hashlib,json
from pathlib import Path

def ref(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True);ap.add_argument('--selector-source',required=True);ap.add_argument('--output-root',required=True);a=ap.parse_args()
 R=Path(a.runs);D=Path(a.output_root);D.mkdir(parents=True,exist_ok=True);source=Path(__file__).with_name('build_pool.py')
 original=R/'ontology_rules/dev16-multi/source/pred.manifest.json';m=json.loads(original.read_text());cfg=m['config'];assert cfg['crf_confidence'] is None
 caches={}
 for kind,folder in [('pubmed','public-phenotagger-dev16-cache'),('bioformer','public-phenotagger-bioformer-dev16-cache'),('biobert','public-phenotagger-biobert-dev16-cache')]:
  c=R/'research'/folder/'terminal/native-dev16-final-cache';caches[kind]={k:ref(c/n) for k,n in [('manifest','manifest.json'),('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]}
 p=dict(execution_ready=True,mode='frozen_dev_input_only_reconstruction',source_sha256=ref(source)['sha256'],documents=16,fixed_source_root=str(Path(a.selector_source).resolve()),input=ref(cfg['input']),base=ref(cfg['base']),sap_candidates=ref(cfg['candidates']),crf=ref(cfg['crf']),crf_confidence_policy='none_original_reference51c5',hpo=ref('/workspace/patientphex-data/hp.obo'),train64_origin=ref(original),caches=caches,output_dir=str(D/'dev-reconstructed'),comparison_only={
 'final_pool':ref(R/'ontology_rules/public-biobert-dev-transfer/predicted/input.pool.jsonl.gz'),
 'bioformer_scores':ref(R/'sequence_crf/public-bioformer-dev-transfer/fitted/selector-scores.jsonl'),
 'pubmed_prepared':ref(R/'sequence_crf/public-selector-dev-transfer/prepared/candidates.jsonl'),
 'bioformer_prepared':ref(R/'sequence_crf/public-bioformer-dev-transfer/prepared/candidates.jsonl')},additional_sources=[ref(original)])
 assert p['comparison_only']['final_pool']['sha256']=='18ac347e1760d960d760de873fe16882525bcee937e2ceceb4a03b163c3c4ee9'
 with (D/'dev-plan.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(json.dumps(ref(D/'dev-plan.json')))
if __name__=='__main__':main()
