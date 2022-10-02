from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss



class Yeji_Model(PreTrainedModel):
    def __init__(self, args):
        super().__init__(args)
        self.num_labels = 128
        # config = AutoConfig.from_pretrained("monologg/kobert")
        self.bert = AutoModel.from_pretrained("monologg/kobert")
        self.dropout = nn.Dropout(0.1)
        self.classifier1 = nn.Linear(786, 393)
        self.classifier2 = nn.Linear(393, 128)

        self.init_weights()

    def forward(self,
                cat2,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                return_dict=None,
                labels=None,
                before_labels=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        sequence_output = self.dropout(outputs[0])
        sequence_output = self.pre_classifier(sequence_output)

        # add before_labels at the sequence_output
        # emb_type_labels = self.type_embedding(before_labels)

        # prediction class by using [ClS] token
        logits = self.classifier1(torch.cat((sequence_output[:, 0, :], cat2), dim=-1))
        logits = self.classifier2(logits)
        loss = None
        if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
