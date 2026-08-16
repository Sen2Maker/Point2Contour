import math
import torch
import torch.nn as nn
import numpy as np


def init_weights_relu(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.0,
            nonlinearity="relu",
            mode="fan_in"
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_lrelu(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight,
            a=1e-2,
            mode="fan_in"
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_selu(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.0,
            nonlinearity="linear",
            mode="fan_in"
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_sigmoid(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_activation_with_init(activation_type):
    if activation_type == "relu":
        return nn.ReLU(inplace=True), init_weights_relu

    if activation_type == "lrelu":
        return nn.LeakyReLU(negative_slope=1e-2, inplace=True), init_weights_lrelu

    if activation_type == "selu":
        return nn.SELU(inplace=True), init_weights_selu

    if activation_type == "sigmoid":
        return nn.Sigmoid(), init_weights_sigmoid

    raise NotImplementedError(f"Unknown activation_type: {activation_type}")


class FourierPosEncoding(nn.Module):

    def __init__(
        self,
        d_in=3,
        num_freqs=6,
        freq_factor=np.pi,
        include_input=True,
    ):
        super().__init__()

        self.d_in = d_in
        self.num_freqs = num_freqs
        self.include_input = include_input

        freqs = freq_factor * (2.0 ** torch.arange(num_freqs))
        self.register_buffer(
            "freqs",
            torch.repeat_interleave(freqs, 2).view(1, -1, 1),
            persistent=False,
        )

        phases = torch.zeros(2 * num_freqs)
        phases[1::2] = np.pi * 0.5
        self.register_buffer(
            "phases",
            phases.view(1, -1, 1),
            persistent=False,
        )

        self.d_out = 2 * num_freqs * d_in
        if include_input:
            self.d_out += d_in

    def forward(self, x):
        original_shape = list(x.shape)

        x_flat = x.reshape(-1, original_shape[-1])

        encoded = x_flat.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)

        encoded = torch.sin(encoded * self.freqs + self.phases)

        original_shape[-1] = -1
        encoded = encoded.reshape(original_shape)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class MLP(nn.Module):

    def __init__(
        self,
        size,
        activation_type="relu",
        bias=True,
        num_pos_encoding=-1,
        include_input=True,
    ):
        super().__init__()

        size = list(size)

        self.original_size = list(size)
        self.activation_type = activation_type
        self.bias = bias
        self.num_pos_encoding = num_pos_encoding

        activation, weights_init = get_activation_with_init(activation_type)

        self.use_pos_encoding = num_pos_encoding is not None and num_pos_encoding > 0

        if self.use_pos_encoding:
            self.pos_encoder = FourierPosEncoding(
                d_in=size[0],
                num_freqs=num_pos_encoding,
                include_input=include_input,
            )
            size[0] = self.pos_encoder.d_out
        else:
            self.pos_encoder = None

        layers = []

        for i in range(len(size) - 1):
            in_dim = size[i]
            out_dim = size[i + 1]

            layers.append(nn.Linear(in_dim, out_dim, bias=bias))

            if i < len(size) - 2:
                layers.append(activation)

        self.mlp = nn.Sequential(*layers)

        self.mlp.apply(weights_init)

    def forward(self, x):
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        return self.mlp(x)

    def forward_simple(self, x):

        return self.forward(x)

    def layer_feature(self, x, k):
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        return self.mlp[:k](x)


if __name__ == "__main__":
    x = torch.rand(10, 3)

    mlp_plain = MLP([3, 4, 5])
    y_plain = mlp_plain(x)

    print("Plain MLP output:", tuple(y_plain.shape))

    mlp_pe = MLP([3, 4, 5], num_pos_encoding=6)
    y_pe = mlp_pe(x)

    print("Fourier MLP output:", tuple(y_pe.shape))
