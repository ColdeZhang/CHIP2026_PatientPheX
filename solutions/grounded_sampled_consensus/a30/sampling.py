"""The single frozen sampling recipe; no model imports or quality-dependent choices."""
import hashlib
SEEDS=(20261024,20261025)
GENERATION={'do_sample':True,'temperature':0.7,'top_p':0.8,'top_k':20,'min_p':0.0,
            'repetition_penalty':1.0,'use_cache':True,'num_return_sequences':1,'num_beams':1,
            'max_new_tokens':1600}
# Presence penalty zero is exactly no added presence LogitsProcessor. It is not
# passed as an unsupported model kwarg. Keep the original native JSON behavior.
def row_seed(seed,row_id):
    assert seed in SEEDS and isinstance(row_id,str) and row_id
    return int.from_bytes(hashlib.sha256(f'{seed}|{row_id}'.encode()).digest()[:8],'big') & ((1<<63)-1)
