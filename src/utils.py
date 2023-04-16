import torch as t
import torchmetrics as tm
from tqdm import tqdm
from datetime import datetime

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


def train(train_loader,
          val_loader,
          model,
          optimizer,
          lf,
          epochs=10,
          device='cpu',
          sm=None,
          verbose=1,
          save_on=None,
          save_params=None
          ):
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

        if verbose == 1:
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
            sm.add_scalar('Train/accuracy', accuracy, epoch)
            sm.add_scalar('Train/precision', precision, epoch)
            sm.add_scalar('Train/recall', recall, epoch)
            sm.add_scalar('Train/loss', epoch_loss, epoch)

            sm.add_scalar('Val/accuracy', eval_loss['accuracy'], epoch)
            sm.add_scalar('Val/precision', eval_loss['precision'], epoch)
            sm.add_scalar('Val/recall', eval_loss['recall'], epoch)
            sm.add_scalar('Val/loss', eval_loss['loss'], epoch)

        if save_on is not None:
            if (epoch + 1) % save_on == 0:
                save_model(model, save_params, eval_loss, postfix=f'epoch_{epoch}_val')
                print('model saved')

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
        'precision' : pre / batch_counter,
        'recall' : rec / batch_counter,
    }



def save_model(model, params, test_results, path='./models/', postfix=''):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    to_save = {
        'model' : model,
        'params' : params,
        'scores' : test_results
    }
    t.save(to_save, path + 'model_' + timestamp + '_' + postfix)
