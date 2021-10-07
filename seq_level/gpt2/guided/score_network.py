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
    elif model_type == 'simple_mlp_w_residual':
        score_network = ScoreNetworkSimpleMLPwReLUResidual(input_size, args)
    elif model_type == 'simple_mlp_w_layer_norm':
        score_network = ScoreNetworkSimpleMLPwLayerNorm(input_size, args)
    elif model_type == 'simple_mlp_w_predictions_v2':
        score_network = ScoreNetworkSimpleMLPwPrediction(input_size, args)
    elif model_type == 'simple_mlp_w_targets':
        score_network = ScoreNetworkSimpleMLPwTargets(input_size, args)
    elif model_type == 'simple_mlp_w_targets_v2':
        score_network = ScoreNetworkSimpleMLPwTargetsV2(input_size, args)
    elif model_type == 'simple_mlp_w_predictions_targets_v2':
        score_network = ScoreNetworkSimpleMLPwPredictionTarget(input_size, args)
    elif model_type == 'simple_mlp_w_targets_v3':
        score_network =ScoreNetworkSimpleMLPwTargetsV3(input_size, args)
    elif model_type == 'simple_mlp_w_targets_v4':
        score_network = ScoreNetworkSimpleMLPwTargetsV4(input_size, args)
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
        self.train_on_predictions = False
        self.train_on_targets = False

    def _get_model_output(self, model, input, target=None,
                          predictions=None, return_cls_token=True):
        pad = model.config.pad_token_id
        device = input.device
        mask = input.ne(pad).float().to(device=device)
        input_ = deepcopy(input).to(device=device)
        input_[input == pad] = 0
        with torch.no_grad():
            model_output = model(input_,
                                attention_mask=mask,
                                output_hidden_states=True)
        if return_cls_token:
            return model_output\
                    .hidden_states[-1][:, -1]
        return model_output

    def _get_input(self, batch, predictions=None):
        if self.train_on_predictions:
            return predictions
        if self.train_on_targets:
            inp, target = batch[:, :-1], batch[:, 1:]
            return inp
        context_batch, _ = utils.wrap_context_batch(batch, self.context_length)
        return context_batch

    def forward(self, model, batch):
        raise NotImplementedError()

class ScoreNetworkSimpleMLP(ScoreNetworkBase):
    def __init__(self, input_size, args, 
                    hidden_size=None,
                    output_size=None,
                    layer_norm_hidden=nn.Identity(), 
                    layer_norm_input=nn.Identity()):
        super(ScoreNetworkSimpleMLP, self).__init__(input_size, args)
        self.hidden_size = hidden_size or args.score_network_hidden_size
        self.output_size = output_size or self.hidden_size

        self.dropout = nn.Dropout(args.score_network_dropout_ratio)
        self.context_length = args.context_length
        self.num_layers = args.score_network_num_layers

        self.input_layer = nn.Sequential(layer_norm_input,
                                nn.Linear(self.input_size, self.hidden_size),
                                self.dropout,)

        self.fc = nn.Sequential(layer_norm_hidden,
                        nn.Linear(self.hidden_size, self.hidden_size),
                        self.dropout,)

        self.output_layer = nn.Sequential(
                                nn.Linear(self.output_size, 1),
                                self.dropout,)

    def forward(self, model, batch, predictions=None):
        inp, targets = batch[:, :-1], batch[:, 1:]

        input = self._get_input(batch, predictions)
        emb = self._get_model_output(model, input, targets=inp, 
                                        predictions=predictions)
        emb = self.input_layer(emb)
        emb = self._apply_hidden_layers(emb)
        output = self.output_layer(emb.view(-1, self.output_size))
        return F.softplus(output)

    def _apply_hidden_layers(self, emb):
        for _ in range(self.num_layers):
            emb = self.fc(emb)
        return emb

class ScoreNetworkSimpleMLPwReLU(ScoreNetworkSimpleMLP):
    def __init__(self, input_size, args, hidden_size=None, output_size=None):
        super(ScoreNetworkSimpleMLPwReLU, self).__init__(input_size, args, 
                                                    hidden_size=hidden_size,
                                                    output_size=output_size)
        old_fc = self.fc

        self.fc = nn.Sequential(
            old_fc,
            nn.ReLU(),
        )

class ScoreNetworkSimpleMLPwReLUResidual(ScoreNetworkSimpleMLPwReLU):
    def _apply_hidden_layers(self, model_output_emb):
        emb = model_output_emb
        for _ in range(self.num_layers):
            emb = emb + self.fc(emb)
        return emb

class ScoreNetworkSimpleMLPwLayerNorm(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args, hidden_size=None, output_size=None):
        super(ScoreNetworkSimpleMLPwReLU, self).__init__(input_size, args, 
                        hidden_size=hidden_size,
                        output_size=output_size,
                        layer_norm_input=nn.LayerNorm(input_size),
                        layer_norm_hidden = nn.LayerNorm(hidden_size or args.score_network_hidden_size))


class ScoreNetworkSimpleMLPwPrediction(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args, hidden_size=None, output_size=None):
        super(ScoreNetworkSimpleMLPwPrediction, self).__init__(input_size, args, 
                                                            hidden_size=hidden_size,
                                                            output_size=output_size)
    def _get_model_output(self, model, input, targets=None,
                          predictions=None, return_cls_token=True):
        emb =  super()._get_model_output(model, predictions)
        return emb


class ScoreNetworkSimpleMLPwTargets(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args, hidden_size=None, output_size=None):
        super(ScoreNetworkSimpleMLPwTargets, self).__init__(input_size, args)
    
    def _get_model_output(self, model, input, targets=None,
                          predictions=None, return_cls_token=True):
        emb =  super()._get_model_output(model, targets)
        return emb


class ScoreNetworkSimpleMLPwTargetsV2(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args, hidden_size=None, output_size=None):
        super(ScoreNetworkSimpleMLPwTargetsV2, self).__init__(input_size * 2, args)

    def _get_model_output(self, model, input, targets=None,
                          predictions=None, return_cls_token=True):
        inp_emb = super()._get_model_output(model, input)
        tgt_emb =  super()._get_model_output(model, targets)
        emb = torch.cat([inp_emb, tgt_emb], dim=-1)
        return emb


class ScoreNetworkSimpleMLPwPredictionTarget(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwPredictionTarget, self).__init__(input_size * 2, args)
        self.train_on_targets = True
    def _get_model_output(self, model, input, targets=None, 
                            predictions=None, return_cls_token=True):
        pred_emb =  super()._get_model_output(model, predictions)
        tgt_emb = super()._get_model_output(model, targets)
        emb = torch.cat([pred_emb, tgt_emb], dim=-1)
        return emb


class ScoreNetworkSimpleMLPwCompleteContext(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwCompleteContext, self).__init__(
                                    input_size=input_size * args.context_length,
                                    args=args)

    def _get_model_output(self, model, input, targets=None, 
                            predictions=None, return_cls_token=True):
        batch_size = input.size(0)
        model_output =  super()._get_model_output(model, input,
                                                  return_cls_token=False)
        emb = model_output\
                .hidden_states[-1].view(batch_size, -1)
        return emb


class ScoreNetworkSimpleMLPwCompleteContextV2(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwCompleteContextV2, self).__init__(
                args=args, input_size=input_size, 
                output_size=args.score_network_hidden_size * args.context_length)

    def _get_model_output(self, model, input, targets=None,
                            predictions=None, return_cls_token=True):
        model_output =  super()._get_model_output(model, input, 
                                                  return_cls_token=False)
        emb = model_output\
                .hidden_states[-1].view(-1, self.input_size)
        return emb


class ScoreNetworkSimpleMLPwTargetsV3(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args, hidden_size=None, output_size=None):
        super(ScoreNetworkSimpleMLPwTargetsV3, self).__init__(input_size, args,
                    output_size=args.score_network_hidden_size * (args.context_length + 1))


    def _get_model_output(self, model, input, targets=None,
                          predictions=None, return_cls_token=True):
        tgt_emb =  super()._get_model_output(model, targets)
        model_output =  super()._get_model_output(model, input, 
                                                  return_cls_token=False)
        inp_emb = model_output\
                    .hidden_states[-1].view(-1, self.input_size)\
                    .view(-1, self.input_size)
        emb = torch.cat([inp_emb, tgt_emb], dim=0)
        return emb

class ScoreNetworkSimpleMLPwTargetsV4(ScoreNetworkSimpleMLPwReLUResidual):
    def __init__(self, input_size, args):
        super(ScoreNetworkSimpleMLPwTargetsV4, self).__init__(
                                    input_size=input_size * (args.context_length + 1),
                                    args=args)

    def _get_model_output(self, model, input, targets=None, 
                            predictions=None, return_cls_token=True):
        batch_size = input.size(0)
        tgt_emb =  super()._get_model_output(model, targets)
        model_output =  super()._get_model_output(model, input,
                                                  return_cls_token=False)
        inp_emb = model_output\
                .hidden_states[-1].view(batch_size, -1)
        emb = torch.cat([inp_emb, tgt_emb], dim=1)
        return emb

def add_args(parser):
    parser.add_argument(
        "--score-network-hidden-size", type=int, default=1024,
    )
    parser.add_argument(
        "--score-network-num-layers", type=int, default=3,
    )
    parser.add_argument(
        "--score-network-dropout-ratio", type=float, default=0.2,
    )
    parser.add_argument(
        "--score-network-type", type=str, default="simple_mlp_complete_context",
    )
    parser.add_argument(
        "--score-network-train-on-predictions", action='store_true',
    )
    parser.add_argument(
        "--score-network-train-on-predictions-targets", action='store_true',
    )
    return parser
