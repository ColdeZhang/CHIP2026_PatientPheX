"""Reuse previously audited published-tokenizer renderer and causal encoding.
No model/Transformers import. New strings still require native runtime replay.
"""
import hashlib
from pathlib import Path

class FrozenTokenizer:
    def __init__(self, tokenizer_path, template_path):
        import tokenizers
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        assert tokenizers.__version__ == "0.23.1" and jinja2.__version__ == "3.1.6"
        self.backend = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def fail(message):
            raise ValueError(message)
        env.globals["raise_exception"] = fail
        self.template = env.from_string(Path(template_path).read_text())
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self.backend.token_to_id(self.eos_token)
        assert self.eos_token_id == 248046

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking, return_dict=False):
        assert enable_thinking is False and return_dict is False
        text = self.template.render(messages=messages, tools=None,
                                    add_generation_prompt=add_generation_prompt, enable_thinking=False)
        return self.backend.encode(text, add_special_tokens=False).ids if tokenize else text

def encode(row, tokenizer):
    kwargs = dict(enable_thinking=False)
    prefix_text = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                               add_generation_prompt=True, **kwargs)
    full_messages = row['messages'] + [{'role':'assistant','content':row['answer']}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False,
                                             add_generation_prompt=False, **kwargs)
    if not full_text.startswith(prefix_text):
        raise ValueError('Rendered assistant prefix mismatch: '+row['row_id'])
    eos = tokenizer.eos_token
    if not eos or full_text[len(prefix_text):] != row['answer'].strip()+eos+'\n':
        raise ValueError('Completion is not exact answer plus chat EOS/newline: '+row['row_id'])
    prefix = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                          add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(full_messages, tokenize=True, return_dict=False,
                                        add_generation_prompt=False, **kwargs)
    if not isinstance(full, list) or not all(isinstance(x,int) for x in full):
        raise ValueError('Unexpected tokenizer output type')
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('Token prefix mismatch or empty completion: '+row['row_id'])
    suffix = full[len(prefix):]
    if suffix.count(tokenizer.eos_token_id) != 1:
        raise ValueError('Supervised answer must contain exactly one chat EOS: '+row['row_id'])
    labels = [-100]*len(prefix) + suffix
    if labels[:len(prefix)] != [-100]*len(prefix) or labels[len(prefix):] != full[len(prefix):]:
        raise ValueError('Assistant-only loss mask is incorrect')
    if not any(x != -100 for x in labels[1:]):
        raise ValueError('No causally supervised tokens')
    return {'row_id':row['row_id'], 'input_ids':full, 'labels':labels,
            'prefix_tokens':len(prefix), 'supervised_tokens':len(suffix),
            'eos_token_id':tokenizer.eos_token_id,
            'eos_supervised_position':len(prefix)+suffix.index(tokenizer.eos_token_id),
            'rendered_prefix_sha256':hashlib.sha256(prefix_text.encode()).hexdigest(),
            'rendered_full_sha256':hashlib.sha256(full_text.encode()).hexdigest()}
