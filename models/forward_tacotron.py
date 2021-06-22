from pathlib import Path
from typing import Union, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, BatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from utils.text.symbols import phonemes

MEL_PAD_VALUE = -11.5129


class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dur_hat):

        dur_hat = dur_hat.detach()

        bs = dur_hat.shape[0]

        ends = torch.cumsum(dur_hat, dim=1)
        mids = ends - dur_hat / 2.

        device = x.device
        seq_len = mids.shape[1]
        mel_len = ends.max().long().item()

        t_range = torch.arange(0, mel_len).long().to(device)
        t_range = t_range.unsqueeze(0)
        t_range = t_range.expand(bs, mel_len)
        t_range = t_range.unsqueeze(-1)
        t_range = t_range.expand(bs, mel_len, seq_len)

        mids = mids.unsqueeze(1)

        diff = t_range - mids
        logits = -(diff ** 2) * 0.1 - 1e-9
        weights = torch.softmax(logits, dim=2)
        x = torch.einsum('bij,bjk->bik', weights, x)

        return x


class SeriesPredictor(nn.Module):

    def __init__(self, num_chars: int, emb_dim=64, conv_dims=256, kernel_size=5,
                 conv_layers=3, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.conv_gru = ConvGru(in_dims=emb_dim, conv_layers=conv_layers, conv_dims=conv_dims,
                                kernel_size=kernel_size, gru_dims=rnn_dims,
                                dropout=dropout)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self,
                x: torch.tensor,
                x_lens: torch.tensor = None,
                alpha=1.0) -> torch.tensor:
        x = self.embedding(x)
        x = self.conv_gru(x, x_lens=x_lens)
        x = self.lin(x)
        return x / alpha


class ConvGru(nn.Module):

    def __init__(self,
                 in_dims: int,
                 conv_layers=3,
                 conv_dims=512,
                 gru_dims=512,
                 kernel_size=5,
                 dropout=0.,
                 padding_value=0) -> None:
        super().__init__()
        self.first_conv = BatchNormConv(in_dims, conv_dims, kernel_size, activation=torch.relu)
        self.last_conv = BatchNormConv(conv_dims, conv_dims, kernel_size, activation=None)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(conv_dims, conv_dims, kernel_size, activation=torch.relu) for _ in range(conv_layers - 2)
        ])
        self.gru = nn.GRU(conv_dims, gru_dims, batch_first=True, bidirectional=True)
        self.dropout = dropout
        self.padding_value = padding_value

    def forward(self,
                x: torch.tensor,
                x_lens: torch.tensor = None) -> torch.tensor:
        x = x.transpose(1, 2)
        x = self.first_conv(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_conv(x)
        x = x.transpose(1, 2)
        if x_lens is not None:
            x = pack_padded_sequence(x, lengths=x_lens, batch_first=True,
                                     enforce_sorted=False)
        x, _ = self.gru(x)
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, padding_value=self.padding_value, batch_first=True)
        return x


class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: int, activation=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        x = self.bnorm(x)
        return x


class ForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 series_embed_dims: int,
                 series_conv_layers: int,
                 series_kernel_size: int,
                 num_chars: int,
                 durpred_conv_dims: int,
                 durpred_rnn_dims: int,
                 durpred_dropout: float,
                 pitch_conv_dims: int,
                 pitch_rnn_dims: int,
                 pitch_dropout: float,
                 pitch_emb_dims: int,
                 pitch_proj_dropout: float,
                 energy_conv_dims: int,
                 energy_rnn_dims: int,
                 energy_dropout: float,
                 energy_emb_dims: int,
                 energy_proj_dropout: float,

                 prenet_kernel_size: int,
                 prenet_conv_layers: int,
                 prenet_conv_dims: int,
                 prenet_gru_dims: int,
                 prenet_dropout: int,
                 main_kernel_size: int,
                 main_conv_layers: int,
                 main_conv_dims: int,
                 main_gru_dims: int,
                 main_dropout: int,
                 postnet_kernel_size: int,
                 postnet_conv_layers: int,
                 postnet_conv_dims: int,
                 postnet_gru_dims: int,
                 postnet_dropout: int,
                 n_mels: int):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = SeriesPredictor(num_chars=num_chars,
                                        emb_dim=series_embed_dims,
                                        conv_layers=series_conv_layers,
                                        kernel_size=series_kernel_size,
                                        conv_dims=durpred_conv_dims,
                                        rnn_dims=durpred_rnn_dims,
                                        dropout=durpred_dropout)
        self.pitch_pred = SeriesPredictor(num_chars=num_chars,
                                          emb_dim=series_embed_dims,
                                          conv_layers=series_conv_layers,
                                          kernel_size=series_kernel_size,
                                          conv_dims=pitch_conv_dims,
                                          rnn_dims=pitch_rnn_dims,
                                          dropout=pitch_dropout)
        self.energy_pred = SeriesPredictor(num_chars=num_chars,
                                           emb_dim=series_embed_dims,
                                           conv_layers=series_conv_layers,
                                           kernel_size=series_kernel_size,
                                           conv_dims=energy_conv_dims,
                                           rnn_dims=energy_rnn_dims,
                                           dropout=energy_dropout)
        self.prenet = ConvGru(in_dims=embed_dims, gru_dims=prenet_gru_dims,
                              conv_layers=prenet_conv_layers, kernel_size=prenet_kernel_size,
                              conv_dims=prenet_conv_dims, dropout=prenet_dropout)
        self.main_net = ConvGru(in_dims=2 * prenet_gru_dims + pitch_emb_dims + energy_emb_dims,
                                gru_dims=main_gru_dims,
                                conv_layers=main_conv_layers, kernel_size=main_kernel_size,
                                conv_dims=main_conv_dims, dropout=main_dropout)
        self.postnet = ConvGru(in_dims=n_mels, gru_dims=postnet_gru_dims,
                               conv_layers=postnet_conv_layers, kernel_size=postnet_kernel_size,
                               conv_dims=postnet_conv_dims, dropout=postnet_dropout)

        self.lin = torch.nn.Linear(2 * main_gru_dims, n_mels)
        self.post_proj = nn.Linear(2*postnet_gru_dims, n_mels, bias=False)
        self.pitch_emb_dims = pitch_emb_dims
        self.energy_emb_dims = energy_emb_dims
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

        if pitch_emb_dims > 0:
            self.pitch_proj = nn.Sequential(
                nn.Conv1d(1, pitch_emb_dims, kernel_size=3, padding=1),
                nn.Dropout(pitch_proj_dropout))
        if energy_emb_dims > 0:
            self.energy_proj = nn.Sequential(
                nn.Conv1d(1, energy_emb_dims, kernel_size=3, padding=1),
                nn.Dropout(energy_proj_dropout))

    def forward(self, batch: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        x_lens = batch['x_len'].cpu()
        pitch = batch['pitch'].unsqueeze(1)
        energy = batch['energy'].unsqueeze(1)

        if self.training:
            self.step += 1

        dur_hat = self.dur_pred(x, x_lens=x_lens).squeeze(-1)
        pitch_hat = self.pitch_pred(x, x_lens=x_lens).transpose(1, 2)
        energy_hat = self.energy_pred(x, x_lens=x_lens).transpose(1, 2)

        x = self.embedding(x)
        x = self.prenet(x)

        if self.pitch_emb_dims > 0:
            pitch_proj = self.pitch_proj(pitch)
            pitch_proj = pitch_proj.transpose(1, 2)
            x = torch.cat([x, pitch_proj], dim=-1)

        if self.energy_emb_dims > 0:
            energy_proj = self.energy_proj(energy)
            energy_proj = energy_proj.transpose(1, 2)
            x = torch.cat([x, energy_proj], dim=-1)

        x = self.lr(x, dur)
        x = self.main_net(x)
        x = self.lin(x)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x + x_post

        x = x.transpose(1, 2)
        x_post = x_post.transpose(1, 2)
        x_post = self.pad(x_post, mel.size(2))
        x = self.pad(x, mel.size(2))

        return {'mel': x, 'mel_post': x_post,
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,
                 energy_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,

                 ) -> Dict[str, np.array]:
        self.eval()

        dur = self.dur_pred(x, alpha=alpha)
        dur = dur.squeeze(2)

        # Fixing breaking synth of silent texts
        if torch.sum(dur) <= 0:
            dur = torch.full(x.size(), fill_value=2, device=x.device)

        pitch_hat = self.pitch_pred(x).transpose(1, 2)
        pitch_hat = pitch_function(pitch_hat)

        energy_hat = self.energy_pred(x).transpose(1, 2)
        energy_hat = energy_function(energy_hat)

        x = self.embedding(x)
        x = self.prenet(x)

        if self.pitch_emb_dims > 0:
            pitch_hat_proj = self.pitch_proj(pitch_hat).transpose(1, 2)
            x = torch.cat([x, pitch_hat_proj], dim=-1)

        if self.energy_emb_dims > 0:
            energy_hat_proj = self.energy_proj(energy_hat).transpose(1, 2)
            x = torch.cat([x, energy_hat_proj], dim=-1)

        x = self.lr(x, dur)
        x = self.main_net(x)
        x = self.lin(x)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x + x_post

        x = x.transpose(1, 2)
        x_post = x_post.transpose(1, 2)

        x, x_post, dur = x.squeeze(), x_post.squeeze(), dur.squeeze()
        x = x.cpu().data.numpy()
        x_post = x_post.cpu().data.numpy()
        dur = dur.cpu().data.numpy()

        return {'mel': x, 'mel_post': x_post, 'dur': dur,
                'pitch': pitch_hat, 'energy': energy_hat}

    def pad(self, x: torch.tensor, max_len: int) -> torch.tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', MEL_PAD_VALUE)
        return x

    def get_step(self) -> int:
        return self.step.data.item()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ForwardTacotron':
        model_config = config['forward_tacotron']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        return ForwardTacotron(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'ForwardTacotron':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = ForwardTacotron.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model