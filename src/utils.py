import torch as t
import torchmetrics as tm
from tqdm import tqdm

class GetMetrics:
    def __init__(self, params = {
        'task' : 'multiclass', 
        'num_classes' : 3, 
        'average' : 'weighted'
    }):
        self.estimators = {
            'accuracy' : tm.Accuracy(**params),
            'precision' : tm.Precision(**params),
            'recall' : tm.Recall(**params)
        }
    
    def __call__(self, preds, labels, metrics = ['accuracy', 'precision', 'recall']):
        out = {}
        for n in metrics:
            if n in self.estimators.keys():
                out[n] = self.estimators[n] ( preds, labels)
            else: print("No {} metric found".format(n))
            
        return out


def train(train_loader, val_loader, model, optimizer, lf, epochs=10, device='cpu', sm=None):
    metrics = GetMetrics()

    
    bar = tqdm(range(epochs))
    epoch_loss = -1.0
    eval_loss = {
        'loss' : -1.0,
        'accuracy' : -1.0,
        'precision' : -1.0,
        'recall' : -1.0,
    }
    accuracy = -1.0
    precision = -1.0
    recall = -1.0
    for epoch in bar:
        l = 0
        acc = 0
        pre = 0
        rec = 0
        batch_count = 0
        for X, y in train_loader:
            batch_count += 1
            
            optimizer.zero_grad()
            X = X.to(device=device)
            out = model(X).to(device='cpu')
            loss = lf(out, y)
            loss.backward()
            optimizer.step()
            
            l += loss
            m = metrics(out, y)
            acc += m['accuracy']
            pre += m['precision']
            rec += m['recall']

            bar.set_description("train_loss: {:.3f}, train_acc_pre_rec: {:.3f}, {:.3f}, {:.3f}; || eval_loss: {:.3f}, eval_acc_pre_rec {:.3f}, {:.3f}, {:.3f}; || local_loss: {:.3f}"\
                                .format(epoch_loss, accuracy, precision, recall,\
                                        eval_loss['loss'], eval_loss['accuracy'], eval_loss['precision'], eval_loss['recall'],\
                                        loss))

        epoch_loss = l / batch_count
        accuracy = acc / batch_count
        precision = pre / batch_count
        recall = rec / batch_count
        eval_loss = eval(val_loader, model, lf, device)

        if sm is not None:
            sm.add_scalar('Accuracy/train', accuracy, epoch)
            sm.add_scalar('Precision/train', precision, epoch)
            sm.add_scalar('Recall/train', recall, epoch)
            sm.add_scalar('Loss/train', epoch_loss, epoch)

            sm.add_scalar('Accuracy/val', eval_loss['accuracy'], epoch)
            sm.add_scalar('Precision/val', eval_loss['precision'], epoch)
            sm.add_scalar('Recall/val', eval_loss['recall'], epoch)
            sm.add_scalar('Loss/val', eval_loss['loss'], epoch)

def eval(val_loader, model, lf, device='cpu', metrics = None):
    if metrics is None:
        metrics = GetMetrics()
    acc = 0
    pre = 0
    rec = 0
    with t.no_grad():
        loss = 0
        batch_counter = 0
        for X, y in val_loader:
            batch_counter += 1
            l = 0
            out = model(X.to(device=device)).to(device='cpu')
            loss += lf(out, y)

            m = metrics(out, y)
            acc += m['accuracy']
            pre += m['precision']
            rec += m['recall']

    return {
        'loss' : loss / batch_counter,
        'accuracy' : acc / batch_counter,
        'precision' : acc / batch_counter,
        'recall' : acc / batch_counter,
    }

