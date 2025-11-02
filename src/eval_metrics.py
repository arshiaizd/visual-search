from typing import Dict, List

def confusion_per_image(gt, pr, grid=10):
    gt, pr = set(gt), set(pr)
    tp = gt & pr
    fn = gt - pr
    fp = pr - gt
    tn = {(r,c) for r in range(grid) for c in range(grid)} - (gt | pr)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn)

def macro_scores(records: List[Dict]):
    TP = sum(len(r['tp']) for r in records)
    FP = sum(len(r['fp']) for r in records)
    FN = sum(len(r['fn']) for r in records)
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return {"precision":prec,"recall":rec,"f1":f1,"TP":TP,"FP":FP,"FN":FN}
