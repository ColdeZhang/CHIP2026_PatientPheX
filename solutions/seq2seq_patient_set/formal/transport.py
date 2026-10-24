"""Generated IDs use native SP decoding, including native UNK, plus the fixed '<' sentinel."""
def decode_generated(sp, ids):
    pieces=[]
    for token in ids:
        if token==1:break
        if token==32099:pieces.append('<')
        elif 0<=token<sp.get_piece_size():pieces.append(sp.id_to_piece(token))
        else:raise ValueError('Undefined generated extra or padded-vocabulary ID: '+str(token))
    return sp.decode_pieces(pieces)
