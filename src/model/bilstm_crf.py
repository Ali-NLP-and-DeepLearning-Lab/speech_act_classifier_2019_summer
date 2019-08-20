import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataset.vrm_dataset import ignore_index
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, device, num_tag,
                embedding_dim,hidden_dim_enc,hidden_dim_dec,
                num_layers=1, dropout_rate=0):

        super(BiLSTM_CRF, self).__init__()

        self.device = device

        self.input_layer_encoder = nn.Linear(embedding_dim,hidden_dim_enc)

        self.encoder = nn.LSTM(hidden_dim_enc, hidden_dim_enc // 2, batch_first = True,num_layers=num_layers, dropout=dropout_rate,bidirectional=True)

        self.crf = CRF(num_tag,batch_first=True).to(device)

        self.dropout = nn.Dropout(dropout_rate)
        self.input_layer = nn.Linear(hidden_dim_enc,hidden_dim_dec)

        self.decoder = nn.LSTM(hidden_dim_dec, hidden_dim_dec // 2, batch_first = True,
                            num_layers=num_layers, dropout=dropout_rate,bidirectional=True)


        self.hidden2tag = nn.Linear(hidden_dim_dec,num_tag)

        self.num_layers = num_layers
        self.hidden_dim_enc = hidden_dim_enc
        self.hidden_dim_dec = hidden_dim_dec
        self.embedding_dim = embedding_dim

    def init_hidden(self,hidden_dim,batch_size):
        return (torch.randn(2*self.num_layers, batch_size, hidden_dim // 2).to(self.device),
                torch.randn(2*self.num_layers, batch_size,hidden_dim // 2).to(self.device))

    def formatter(self,data):

        utterances, poses, qtags, callers, masks, labels = data
        utt_mask = (labels != ignore_index).to(torch.uint8)
        return [utterances,masks, utt_mask] ,labels

    def forward(self,x):


        utt_mask = x[2]
        mask = x[1]

        # qtags : b * c
        x = x[0]


        b,c,l,ed = x.shape
        x = x.view(b*c*l,ed)
        x = self.input_layer_encoder(x)
        x = F.tanh(x)

        x = x.view(b*c,l,-1)
        #x = self.dropout(x)

        hidden_enc = self.init_hidden(self.hidden_dim_enc,b*c)
        x, hidden_enc = self.encoder(x,hidden_enc)


        x = x*mask.view(b*c,-1).unsqueeze(-1).to(x.dtype)
        x = x.sum(1)
        x = self.dropout(x)

        x = self.input_layer(x)
        x = F.tanh(x)

        x = x.view(b,c,-1)

        hidden_dec = self.init_hidden(self.hidden_dim_dec,b)
        x, hidden_dec = self.decoder(x,hidden_dec)
        x = self.dropout(x)
        # now ready for CRF
        x = x.contiguous()
        x = x.view(b*c,-1)
        #x = self.decoder_proj1(x)
        #x = F.relu(x)
        #x = self.decoder_proj2(x)
        #x = F.relu(x)

        x = self.hidden2tag(x)
        #x = F.softmax(x,-1)
        utt_vecs = x.view(b,c,-1)

        if self.training:

            return utt_vecs,utt_mask #,masks

        else :
            preds = self.crf.decode(utt_vecs,utt_mask) #, masks)
            return preds[0]

    def loss(self,utt_vm,targets):
        utt_vecs, utt_mask = utt_vm

        return -self.crf(utt_vecs,targets,mask=utt_mask,reduction='token_mean')



def build_bilstm_crf(device,config,multiplier=1):
    model = BiLSTM_CRF(device=device,
                num_tag=config.num_tag,
                embedding_dim=config.embedding_dim,
                hidden_dim_enc=config.hidden_dim_enc*multiplier,
                hidden_dim_dec=config.hidden_dim_dec*multiplier,
                num_layers=config.num_layers,
                dropout_rate=config.dropout_rate)
    return model
