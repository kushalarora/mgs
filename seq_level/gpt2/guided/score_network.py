import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch

import seq_level.gpt2.utils as utils

def build_score_network(input_size, args):
    score_network = None
    model_type = args.score_network_type
    if model_type == 'simple_mlp':
        score_network = ScoreNetworkSimpleMLP(input_size, args)
    elif model_type == 'simple_mlp_w_relu':
        score_network = ScoreNetworkSimpleMLPwReLU(input_size, args)
    elif model_type == 'simple_mlp_w_layer_norm':
        score_network = ScoreNetworkSimpleMLPwLayerNorm(input_size, args)
    elif model_type == 'simple_mlp_w_predictions':
        args.score_network_train_on_predictions = True
        score_network = ScoreNetworkSimpleMLPwLayerNorm(input_size, args)
    elif model_type == 'simple_mlp_w_predictions_targets':
        score_network = ScoreNetworkSimpleMLPwPredictionTarget(input_size, args)
    elif model_type == 'simple_mlp_complete_context':
        score_network = ScoreNetworkSimpleMLPwCompleteContext(input_size, args)
    elif model_type == 'simple_mlp_complete_context_v2':
        score_network = ScoreNetworkSimpleMLPwCompleteContextV2(input_size, args)
    else:
        raise ValueError(f"Score Model Type: {model_type} not found!")
    
    if args.use_saved_score_network:
            checkpoint = torch.load(args.score_network_file)
            score_network.load_state_dict(checkpoint['model_save_dict'])
            epochs = score_model_checkpoint['epochs']
            logging.info(f"Loading scorer trained for {epochs} epochs" + \
                            f" from {args.score_network_file}.")
    return score_network
            
class ScoreNetworkBase(nn.Module):
    def __init__(self, input_size, args):

        super(ScoreNetworkBase, self).__init__()
        self.input_size = input_size
        self.train_on_predictions = args.score_network_train_on_predictions

    def _get_model_output(self, model, batch, predictions=None):
        pad = model.config.pad_token_id
        device = batch.device
        mask = batch.ne(pad).float().to(device=device)
        batch_ = deepcopy(batch).to(device=device)
        batch_[batch == pad] = 0
        with torch.no_grad():
            model_output = model(batch_,
                                attention_mask=mask,
                                output_hidden_states=True)
        return model_output

    def _get_input(self, batch, predictions=None):
        if self.train_on_predictions:
            return predictions
        context_batch, _ = utils.wrap_context_batch(
                    batch, self.context_length)
        return context_batch

    def forward(self, model, batch):
        raise NotImplementedError()

class ScoreNetworkSimpleMLP(ScoreNetworkBase):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLP, self).__init__(input_size, args)
        self.hidden_size = args.score_network_hidden_size

        self.dropout = nn.Dropout(args.score_network_dropout_ratio)
        self.context_length = args.context_length
        self.num_layers = args.score_network_num_layers

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            self.dropout,
        )
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, model, batch, predictions=None):
        input = self._get_input(batch, predictions)
        model_output = self._get_model_output(model, input)

        emb = model_output\
                .hidden_states[-1][:, -1]

        emb = self.dropout(self.input_layer(emb))
        for _ in range(self.num_layers):
            emb = self.fc(emb)
        output = self.output_layer(emb)
        return F.softplus(output)

class ScoreNetworkSimpleMLPwReLU(ScoreNetworkSimpleMLP):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwReLU, self).__init__(input_size, args)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            self.dropout,
            nn.ReLU(),
        )

class ScoreNetworkSimpleMLPwLayerNorm(ScoreNetworkSimpleMLPwReLU):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwReLU, self).__init__(input_size, args)
        self.layer_norm_input = nn.LayerNorm(self.input_size)
        self.layer_norm_hidden = nn.LayerNorm(self.hidden_size)

    def forward(self, model, batch, predictions=None):
        input = self._get_input(batch, predictions)
        model_output = self._get_model_output(model, input)

        emb = model_output\
                .hidden_states[-1][:, -1]

        emb = self.layer_norm_input(emb)
        emb = self.dropout(self.input_layer(emb))
        for _ in range(self.num_layers - 1):
            emb = self.layer_norm_hidden(emb + self.fc(emb))
        emb = self.fc(emb)
        output = self.output_layer(emb)
        return F.softplus(output)

class ScoreNetworkSimpleMLPwPredictionTarget(ScoreNetworkSimpleMLPwLayerNorm):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwPredictionTarget, self).__init__(input_size * 2, args)

    def forward(self, model, batch, predictions=None):
        inp, target = batch[:, :-1], batch[:, 1:]
        pred_model_output =  self._get_model_output(model, predictions)
        target_model_output = self._get_model_output(model, inp)

        pred_emb = pred_model_output.hidden_states[-1][:, -1]
        tgt_emb = target_model_output.hidden_states[-1][:, -1]

        emb = torch.cat([pred_emb, tgt_emb], dim=-1)

        emb = self.layer_norm_input(emb)
        emb = self.dropout(self.input_layer(emb))
        for _ in range(self.num_layers - 1):
            emb = self.layer_norm_hidden(self.fc(emb))
        emb = self.fc(emb)
        output = self.output_layer(emb)

        # torch.exp ensures score is in the range (0, +\inf)
        return F.softplus(output)

class ScoreNetworkSimpleMLPwCompleteContext(ScoreNetworkSimpleMLPwLayerNorm):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwCompleteContext, self).__init__(args=args,
                                            input_size=input_size * args.context_length)

    def forward(self, model, batch, predictions=None):
        input = self._get_input(batch, predictions)
        model_output = self._get_model_output(model, input)
        batch_size = batch.size(0)
        emb = model_output\
                .hidden_states[-1].view(batch_size, -1)

        emb = self.layer_norm_input(emb)
        emb = self.dropout(self.input_layer(emb))
        for _ in range(self.num_layers - 1):
            emb = self.layer_norm_hidden(emb + self.fc(emb))
        emb = self.fc(emb)
        output = self.output_layer(emb)
        return F.softplus(output)

class ScoreNetworkSimpleMLPwCompleteContextV2(ScoreNetworkSimpleMLPwLayerNorm):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwCompleteContextV2, self).__init__(args=args,
                                            input_size=input_size)
        self.output_layer = nn.Linear(self.hidden_size * args.context_length, 1)

    def forward(self, model, batch, predictions=None):
        input = self._get_input(batch, predictions)
        model_output = self._get_model_output(model, input)
        batch_size = batch.size(0)
        emb = model_output\
                .hidden_states[-1].view(-1, self.input_size)

        emb = self.layer_norm_input(emb)
        emb = self.dropout(self.input_layer(emb))
        for _ in range(self.num_layers - 1):
            emb = self.layer_norm_hidden(emb + self.fc(emb))
        emb = self.fc(emb)
        output = self.output_layer(emb.view(batch_size, -1))
        return F.softplus(output)

def add_args(parser):
    parser.add_argument(
        "--score-network-hidden-size", type=int, default=1024,
    )
    parser.add_argument(
        "--score-network-num-layers", type=int, default=1,
    )
    parser.add_argument(
        "--score-network-dropout-ratio", type=float, default=0.0,
    )
    parser.add_argument(
        "--score-network-type", type=str, default="simple_mlp",
    )
    parser.add_argument(
        "--score-network-train-on-predictions", action='store_true',
    )
    parser.add_argument(
        "--score-network-train-on-predictions-targets", action='store_true',
    )
    return parser
