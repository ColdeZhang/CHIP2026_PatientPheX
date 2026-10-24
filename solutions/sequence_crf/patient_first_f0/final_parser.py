"""Strict patient-first final-response envelope; never recover JSON fragments."""
from patient_first import invert_patient_answer, validate_input

EOS_IDS = frozenset({248044, 248046})


def parse_final(record, prompt):
    validate_input(prompt)
    if 'answer' in prompt:
        raise ValueError('Inference input contains an answer')
    if record.get('row_id') != prompt['row_id'] or str(record.get('pmc_id')) != str(prompt['pmc_id']):
        raise ValueError('Response identity mismatch')
    tokens = record.get('generated_token_ids')
    if (not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError('Require actual generated token IDs')
    if tokens[-1] not in EOS_IDS or any(token in EOS_IDS for token in tokens[:-1]):
        raise ValueError('Require a single actual terminal EOS')
    if record.get('output_tokens') != len(tokens) or record.get('hit_token_limit') is not False:
        raise ValueError('Token accounting or terminal status differs')
    if record.get('removed_demonstrations') != 0:
        raise ValueError('Prompt removal is forbidden')
    text = record.get('response')
    if not isinstance(text, str):
        raise ValueError('Require the complete new response text')
    return invert_patient_answer(text, list(prompt['code_map']), prompt['patient_ids'])
