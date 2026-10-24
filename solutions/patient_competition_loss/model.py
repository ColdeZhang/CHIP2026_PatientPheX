"""Only the patient objective changes; exact executed geometry model is inherited."""
import torch
from vendor.geometry_model import PatientSpanModel as GeometryModel, optimizer_groups
from vendor.gp import multilabel_loss


def grouped_patient_loss(bag_logits, row, zero_reference):
    """All concepts, all representable patients; masked cells never negatives.

    A concept with exclusively masked bags contributes zero to the mean over C.
    No label or instance is rewritten, and different documents never mix.
    """
    bags = row['bags']
    masked = row.get('masked_bags', [])
    concepts = sorted({tuple(b['concept']) for b in bags + masked})
    if not bags:
        assert bag_logits is None or bag_logits.numel() == 0
        return zero_reference.sum()*0, {'concepts':len(concepts),'valid_cells':0,'masked_cells':len(masked),'all_masked_concepts':len(concepts)}
    assert bag_logits.ndim == 1 and len(bag_logits) == len(bags)
    index = {c:i for i,c in enumerate(concepts)}
    cells = [(index[tuple(b['concept'])], b['patient_index']) for b in bags]
    assert len(cells) == len(set(cells))
    pi = {p['patient_id']:i for i,p in enumerate(row['patients'])}
    masked_cells = {(index[tuple(b['concept'])],pi[b['patient_id']]) for b in masked}
    assert not set(cells)&masked_cells
    shape = (len(concepts),len(row['patients']))
    ci = torch.tensor([c[0] for c in cells],device=bag_logits.device,dtype=torch.long)
    patient = torch.tensor([c[1] for c in cells],device=bag_logits.device,dtype=torch.long)
    logits = bag_logits.new_zeros(shape).index_put((ci,patient),bag_logits)
    positive = torch.zeros(shape,device=bag_logits.device,dtype=torch.bool)
    valid = torch.zeros_like(positive)
    positive[ci,patient] = torch.tensor([bool(b['label']) for b in bags],device=bag_logits.device)
    valid[ci,patient] = True
    return multilabel_loss(logits,positive,valid), {'concepts':len(concepts),'valid_cells':len(cells),'masked_cells':len(masked_cells),'all_masked_concepts':len(concepts)-len({c[0] for c in cells})}


class PatientSpanModel(GeometryModel):
    def loss(self,row,retain_hidden=False):
        info = super().loss(row,retain_hidden)
        # super computes the old BCE for its original interface; no backward is
        # taken through that discarded scalar. The scores and max bags are exact.
        reloss, groups = grouped_patient_loss(info['bag_logits'],row,info['NER_loss'])
        info['bag_loss'] = reloss
        info['loss'] = info['NER_loss'] + reloss
        info['patient_loss_groups'] = groups
        return info
