from src.train.model.utils import *
from tqdm.auto import tqdm
from src.schema.schema import JobArgs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
from loguru import logger
from statistics import mean

# tqdm.pandas()


BASE_MODEL_PATH= './'
MODEL_PATH = "pytorch_model.bin"

class EntityModel(nn.Module):
    def __init__(self, num_tag,job_args: JobArgs):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.job_args= job_args
        self.encoder =  AlbertModel.from_pretrained(BASE_MODEL_PATH)
        #self.rnn = nn.LSTM(768, 768)
        self.drop_out = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
    
    def forward(self, ids, mask, token_type_ids):
        embeds, embeddings= self.encoder(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        bo_tag = self.drop_out(embeds)
        #bo_tag, _ = self.rnn(bo_tag)
        tag = self.out_tag(bo_tag)

        return tag

    def calculate_accuracy(self, targets, predictions):
        return accuracy_score(targets, predictions)

    def calculate_precision(self, targets, predictions):
        return precision_score(targets, predictions, average='macro', zero_division=0)

    def calculate_recall(self, targets, predictions):
        return recall_score(targets, predictions, average='macro', zero_division=0)

    def calculate_f1score(self, targets, predictions):
        return f1_score(targets, predictions, average='macro', zero_division=0)

    def loss_fn(self, output, target, mask, num_labels):
        lfn = nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, num_labels)
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(lfn.ignore_index).type_as(target)
        )
        loss = lfn(active_logits, active_labels)

        del active_loss, active_logits, active_labels, lfn
        torch.cuda.empty_cache()

        return loss

    def get_optimizer(self, sentences, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(
                        nd in n for nd in no_decay
                    )
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(
                        nd in n for nd in no_decay
                    )
                ],
                "weight_decay": 0.0,
            },
        ]


        num_train_steps = int(
            len(sentences) / (self.job_args.model_configs.training_params.batch_size)* (self.job_args.model_configs.training_params.epoch)
        )
        if self.job_args.model_configs.incremental:
            optimizer = torch.optim.AdamW(optimizer_parameters, lr = 1e-7)
        else:
            optimizer = torch.optim.AdamW(optimizer_parameters, lr = self.job_args.model_configs.training_params.initial_lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_train_steps
        )
        return optimizer, scheduler


def train_ner_model(model,optimizer,scheduler,dataset_loader,epochs,epoch_start,save_path):        
#    loss_l=[9999]
#    accuracy_l=[0]
#    m_=[]
    min_loss = sys.maxsize
    for epoch in range(epoch_start,epochs):
        print(epoch)
        model.train()

        loss_epoch, acc_epoch, precision_epoch, recall_epoch, f1score_epoch = ([] for _ in range(5))

        for _,data in enumerate(dataset_loader, 0):

            ids = data['ids'].to(device, dtype = torch.long, non_blocking=True)
            mask = data['mask'].to(device, dtype = torch.long, non_blocking=True)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long, non_blocking=True)
            targets = data['target_tag'].to(device, dtype = torch.long, non_blocking=True)
            outputs = model(ids, mask, token_type_ids)

            loss = model.loss_fn(outputs, targets,mask,num_labels = model.num_tag)
            loss_epoch.append(loss.item())

            mask = mask.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()[mask == 1].tolist()
            outputs = outputs.cpu().detach().numpy()[mask == 1]
            predictions = np.argmax(outputs, axis=1).tolist()

            acc_epoch.append(model.calculate_accuracy(targets, predictions))
            precision_epoch.append(model.calculate_precision(targets, predictions))
            recall_epoch.append(model.calculate_recall(targets, predictions))
            f1score_epoch.append(model.calculate_f1score(targets, predictions))

            # if _%(len(dataset_loader)*0.05)==0:
            #     min_loss = min(loss,min_loss)
            #     if loss == min_loss:
            #         joblib.dump(model,save_path+'/model_best.bin')
#                outputs = (np.array(outputs.cpu().detach().numpy()) > 0.5)*1
# #                 print(outputs)
 #               targets = targets.cpu().detach().numpy()
#                 accuracy = accuracy_score(targets, outputs)
#                if min(loss_l) > loss:
#                    joblib.dump(model,save_path+'/model_best.bin')
                #print('\naccuracy = ',accuracy)
#                 accuracy_l.append(accuracy)
#                 m_.append(metrics_(targets,outputs))
#                 joblib.dump({'loss':loss_l,'metrics':m_},save_path+'/reports.dict')
                # joblib.dump(model,save_path+'/model.bin')
                # joblib.dump(model,save_path+'/ner_model.bin')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            del ids, mask, token_type_ids, outputs, loss
            torch.cuda.empty_cache()

        logger.info(f'Epoch: {epoch}, '
                    f'Loss:  {mean(loss_epoch)}, '
                    f'Accuracy:  {mean(acc_epoch)}, '
                    f'Precision:  {mean(precision_epoch)}, '
                    f'Recall:  {mean(recall_epoch)}, '
                    f'F1-Score:  {mean(f1score_epoch)}')

        mean_loss = mean(loss_epoch)
        if mean_loss < min_loss:
            min_loss = mean_loss
            torch.save(model.state_dict(), save_path + "ner_albert_model.dt")

        joblib.dump(model,save_path+'/model_final_'+str(epoch)+'.bin')
        
    print('Done :)')
    return model
