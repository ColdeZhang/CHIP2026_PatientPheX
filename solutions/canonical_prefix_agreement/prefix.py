"""Read closed string items from the original native-token grammar, never repair JSON."""
import json


def trace(codec,tokens,body,natural_eos):
    decode=lambda xs:codec.tokenizer.decode(xs,skip_special_tokens=False)
    assert decode(tokens)==body+('<|im_end|>' if natural_eos else '')
    payload=tokens[:-1] if natural_eos else tokens
    assert not natural_eos or tokens[-1]==codec.eos_id
    items=[];seen=set();node=codec.root;start=len(codec.open);captured=False;finished=False
    for k,token in enumerate(tokens):
        # This is the original actual callback's legality and used-set rule.
        if token not in codec.allowed(tokens[:k]):raise ValueError('Token outside original grammar')
        if tokens[0]==codec.empty[0] or k<len(codec.open) or token==codec.eos_id:continue
        if finished:raise ValueError('Token after completed list')
        node=node.children[token]
        candidates={name for name,_ in node.descendants if name not in seen}
        if not captured and len(candidates)==1:
            name=next(iter(candidates));literal=json.dumps(name,ensure_ascii=False)
            block=decode(tokens[start:k+1])
            if block.startswith(literal):
                text_start=len(decode(tokens[:start]));text_end=text_start+len(literal)
                assert body[text_start:text_end]==literal
                items.append(dict(name=name,hpo_id=codec.names[name],token_start=start,token_end=k+1,
                    text_start=text_start,text_end=text_end,raw_literal=literal,
                    token_span_is_covering=True,token_span_decoded=block))
                captured=True
        if node.terminal:
            name,continuation=node.terminal
            assert captured and items[-1]['name']==name and name not in seen
            seen.add(name);captured=False;node=codec.root;start=k+1;finished=not continuation
    assert len(items)==len({x['name'] for x in items})
    accepted_text_end=items[-1]['text_end'] if items else 0
    pending_start=min(start,len(payload));pending_char=len(decode(payload[:pending_start]))
    return dict(items=items,hpo_ids=[x['hpo_id'] for x in items],
        natural_eos=natural_eos,complete_json=finished or (payload==codec.empty),
        unaccepted_text_tail=dict(start=accepted_text_end,end=len(body),text=body[accepted_text_end:]),
        grammar_pending_block=dict(token_start=pending_start,token_end=len(payload),text_start=pending_char,
            text_end=len(body),text=body[pending_char:]),
        semantics='Accept original closed JSON string items; list/EOS may remain incomplete; no token/text repair')
