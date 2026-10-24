"""Original G JSON semantics; only native-SP unknown '<' uses existing extra_id_0."""
import json

LITERAL_LESS_THAN = 32099  # Existing <extra_id_0>; no new token or embedding.


def encode(sp, original):
    parsed=json.loads(original)
    assert '<extra_id_0>' not in original
    pieces=sp.encode_as_immutable_proto(original).pieces
    ids=[];mapped=0
    for piece in pieces:
        if piece.id==sp.unk_id():
            if piece.surface!='<':raise ValueError('Unsupported native target token; no silent repair')
            ids.append(LITERAL_LESS_THAN);mapped+=1
        else:ids.append(piece.id)
    ids.append(1)
    restored=decode(sp,ids)
    assert json.loads(restored)==parsed and restored==original
    return ids,original.replace('<','<extra_id_0>'),mapped


def decode(sp, ids):
    pieces=[]
    for token in ids:
        if token==1:break
        if token==LITERAL_LESS_THAN:pieces.append('<')
        elif 0<=token<sp.get_piece_size() and token not in (0,2):pieces.append(sp.id_to_piece(token))
        else:raise ValueError('Invalid or unknown generated transport token')
    return sp.decode_pieces(pieces)
