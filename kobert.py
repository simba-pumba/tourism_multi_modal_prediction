from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel
import torch.nn as nn
import torch

AutoTokenizer.from_pretrained("monologg/kobert")
AutoModel.from_pretrained("monologg/kobert")

class Yeji_Model(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.type_emb_hidden = 16

        self.bert = PreTrainedModel(config)

        self.dropout = nn.Dropout(0.1)
        self.classifier1 = nn.Linear(config.hidden_size, self.type_emb_hidden)
        self.embedding1 = nn.Embedding(self.num_labels, self.type_emb_hidden)
        self.embedding2 = nn.Embedding(self.num_labels, self.type_emb_hidden)
        self.classifier2 = nn.Linear(self.type_emb_hidden*2, self.num_labels)

        self.init_weights()

    def forward(self,
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
        emb_type_labels = self.type_embedding(before_labels)

        # prediction class by using [ClS] token
        logits = self.classifier(torch.cat((sequence_output[:, 0, :], emb_type_labels), dim=-1))

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
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