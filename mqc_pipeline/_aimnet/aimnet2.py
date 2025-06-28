from typing import Dict, List, Mapping, Sequence, Tuple, Union, ClassVar, Final

import torch
from torch import Tensor, nn

from . import nbops, ops
from .aev import AEVSV, ConvSV
from .core import MLP, Embedding


class AIMNet2Base(nn.Module):  # pylint: disable=abstract-method
    """Base class for AIMNet2 models. Implements pre-processing data:
    converting to right dtype and device, setting nb mode, calculating masks.
    """

    __default_dtype = torch.get_default_dtype()

    _required_keys: Final = ["coord", "numbers", "charge"]
    _required_keys_dtype: Final = [
        __default_dtype, torch.int64, __default_dtype
    ]
    _optional_keys: Final = [
        "mult", "nbmat", "nbmat_lr", "mol_idx", "shifts", "shifts_lr", "cell"
    ]
    _optional_keys_dtype: Final = [
        __default_dtype,
        torch.int64,
        torch.int64,
        torch.int64,
        __default_dtype,
        __default_dtype,
        __default_dtype,
    ]
    __constants__: ClassVar = [
        "_required_keys", "_required_keys_dtype", "_optional_keys",
        "_optional_keys_dtype"
    ]

    def __init__(self):
        super().__init__()

    def _prepare_dtype(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for k, d in zip(self._required_keys, self._required_keys_dtype):
            assert k in data, f"Key {k} is required"
            data[k] = data[k].to(d)
        for k, d in zip(self._optional_keys, self._optional_keys_dtype):
            if k in data:
                data[k] = data[k].to(d)
        return data

    def prepare_input(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Some sommon operations"""
        data = self._prepare_dtype(data)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        assert data["charge"].ndim == 1, "Charge should be 1D tensor."
        if "mult" in data:
            assert data["mult"].ndim == 1, "Mult should be 1D tensor."
        return data


class AIMNet2(AIMNet2Base):

    def __init__(
            self,
            aev: Dict,
            nfeature: int,
            d2features: bool,
            ncomb_v: int,
            hidden: Tuple[List[int]],
            aim_size: int,
            outputs: Union[List[nn.Module], Dict[str, nn.Module]],
            num_charge_channels: int = 1,
            extra_passes: int = 1  # Number of additional passes
    ):
        super().__init__()

        if num_charge_channels not in [1, 2]:
            raise ValueError(
                "num_charge_channels must be 1 (closed shell) or 2 (NSE for open-shell)."
            )
        self.num_charge_channels = num_charge_channels
        self.extra_passes = extra_passes

        self.aev = AEVSV(**aev)
        nshifts_s = aev["nshifts_s"]
        nshifts_v = aev.get("nshitfs_v") or nshifts_s
        if d2features:
            if nshifts_s != nshifts_v:
                raise ValueError(
                    "nshifts_s must be equal to nshifts_v for d2features")
            nfeature_tot = nshifts_s * nfeature
        else:
            nfeature_tot = nfeature
        self.nfeature = nfeature
        self.nshifts_s = nshifts_s
        self.d2features = d2features

        self.afv = Embedding(num_embeddings=64,
                             embedding_dim=nfeature,
                             padding_idx=0)

        with torch.no_grad():
            nn.init.orthogonal_(self.afv.weight[1:])
            if d2features:
                self.afv.weight = nn.Parameter(
                    self.afv.weight.clone().unsqueeze(-1).expand(
                        64, nfeature, nshifts_s).flatten(-2, -1))

        conv_param = {
            "nshifts_s": nshifts_s,
            "nshifts_v": nshifts_v,
            "ncomb_v": ncomb_v,
            "do_vector": True
        }
        self.conv_a = ConvSV(nchannel=nfeature,
                             d2features=d2features,
                             **conv_param)
        self.conv_q = ConvSV(nchannel=num_charge_channels,
                             d2features=False,
                             **conv_param)

        mlp_param = {"activation_fn": nn.GELU(), "last_linear": True}
        mlps = [
            MLP(
                n_in=self.conv_a.output_size() + nfeature_tot,
                n_out=nfeature_tot + 2 * num_charge_channels,
                hidden=hidden[0],
                **mlp_param,
            )
        ]
        mlp_param = {"activation_fn": nn.GELU(), "last_linear": False}
        for h in hidden[1:-1]:
            mlps.append(
                MLP(
                    n_in=self.conv_a.output_size() +
                    self.conv_q.output_size() + nfeature_tot +
                    num_charge_channels,
                    n_out=nfeature_tot + 2 * num_charge_channels,
                    hidden=h,
                    **mlp_param,
                ))
        mlp_param = {"activation_fn": nn.GELU(), "last_linear": False}
        mlps.append(
            MLP(
                n_in=self.conv_a.output_size() + self.conv_q.output_size() +
                nfeature_tot + num_charge_channels,
                n_out=aim_size,
                hidden=hidden[-1],
                **mlp_param,
            ))

        extra_mlps = []
        # For each property type (ESP and Electronic Levels)
        for property_idx in range(2):
            # First MLP - similar to first message passing MLP
            first_mlp_param = {"activation_fn": nn.GELU(), "last_linear": True}
            extra_mlps.append(
                MLP(n_in=aim_size + self.conv_a.output_size() +
                    self.conv_q.output_size() + nfeature_tot +
                    num_charge_channels,
                    n_out=nfeature_tot + 2 * num_charge_channels,
                    hidden=hidden[0],
                    **first_mlp_param))

            # Second MLP - takes updated features and charges
            second_mlp_param = {
                "activation_fn": nn.GELU(),
                "last_linear": True
            }
            extra_mlps.append(
                MLP(n_in=self.conv_a.output_size() +
                    self.conv_q.output_size() + nfeature_tot +
                    num_charge_channels,
                    n_out=nfeature_tot + 2 * num_charge_channels,
                    hidden=hidden[1] if len(hidden) > 1 else hidden[0],
                    **second_mlp_param))

            # Third MLP - takes updated features and charges
            third_mlp_param = {
                "activation_fn": nn.GELU(),
                "last_linear": False
            }
            extra_mlps.append(
                MLP(
                    n_in=self.conv_a.output_size() +
                    self.conv_q.output_size() + nfeature_tot +
                    num_charge_channels,
                    n_out=256,  # Fixed output size for property-specific modules
                    hidden=hidden[-1],
                    **third_mlp_param))

        self.mlps = nn.ModuleList(mlps)
        self.extra_mlps = nn.ModuleList(extra_mlps)

        if isinstance(outputs, Sequence):
            self.outputs = nn.ModuleList(outputs)
        elif isinstance(outputs, Mapping):
            self.outputs = nn.ModuleDict(outputs)
        else:
            raise TypeError("`outputs` is not either list or dict")

    def _preprocess_spin_polarized_charge(
            self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "mult" not in data:
            raise ValueError(
                "mult key is required for NSE if two channels for charge are not provided"
            )
        _half_spin = 0.5 * (data["mult"] - 1.0)
        _half_q = 0.5 * data["charge"]
        data["charge"] = torch.stack(
            [_half_q + _half_spin, _half_q - _half_spin], dim=-1)
        return data

    def _postprocess_spin_polarized_charge(
            self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data["spin_charges"] = data["charges"][..., 0] - data["charges"][...,
                                                                         1]
        data["charges"] = data["charges"].sum(dim=-1)
        data["charge"] = data["charge"].sum(dim=-1)
        return data

    def _prepare_property_in_a(self, prop_data, data):
        a_i, a_j = nbops.get_ij(prop_data["a"], data)
        avf_a = self.conv_a(a_j, data["gs"], data["gv"])
        if self.d2features:
            a_i = a_i.flatten(-2, -1)
        _in = torch.cat([a_i.squeeze(-2), avf_a], dim=-1)
        return _in

    def _prepare_property_in_q(self, prop_data, data):
        q_i, q_j = nbops.get_ij(prop_data["charges"], data)
        avf_q = self.conv_q(q_j, data["gs"], data["gv"])
        _in = torch.cat([q_i.squeeze(-2), avf_q], dim=-1)
        return _in

    def _prepare_in_a(self, data: Dict[str, Tensor]) -> Tensor:
        a_i, a_j = nbops.get_ij(data["a"], data)
        avf_a = self.conv_a(a_j, data["gs"], data["gv"])
        if self.d2features:
            a_i = a_i.flatten(-2, -1)
        _in = torch.cat([a_i.squeeze(-2), avf_a], dim=-1)
        return _in

    def _prepare_in_q(self, data: Dict[str, Tensor]) -> Tensor:
        q_i, q_j = nbops.get_ij(data["charges"], data)
        avf_q = self.conv_q(q_j, data["gs"], data["gv"])
        _in = torch.cat([q_i.squeeze(-2), avf_q], dim=-1)
        return _in

    def _update_q(self,
                  data: Dict[str, Tensor],
                  x: Tensor,
                  delta_q: bool = True) -> Dict[str, Tensor]:
        _q, _f, delta_a = x.split(
            [
                self.num_charge_channels,
                self.num_charge_channels,
                x.shape[-1] - 2 * self.num_charge_channels,
            ],
            dim=-1,
        )
        # for loss
        data["_delta_Q"] = data["charge"] - nbops.mol_sum(_q, data)
        q = data["charges"] + _q if delta_q else _q
        f = _f.pow(2)
        q = ops.nse(data["charge"], q, f, data, epsilon=1.0e-6)
        data["charges"] = q
        data["a"] = data["a"] + delta_a.view_as(data["a"])
        return data

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_input(data)

        # initial features
        a: Tensor = self.afv(data["numbers"])
        if self.d2features:
            a = a.unflatten(-1, (self.nfeature, self.nshifts_s))
        data["a"] = a

        # NSE case
        if self.num_charge_channels == 2:
            data = self._preprocess_spin_polarized_charge(data)
        else:
            # make sure that charge has channel dimension
            data["charge"] = data["charge"].unsqueeze(-1)

        # AEV
        data = self.aev(data)

        # MP iterations
        _npass = len(self.mlps)
        for ipass, mlp in enumerate(self.mlps):
            if ipass == 0:
                _in = self._prepare_in_a(data)
            else:
                _in = torch.cat(
                    [self._prepare_in_a(data),
                     self._prepare_in_q(data)],
                    dim=-1)

            _out = mlp(_in)
            if data["_input_padded"].item():
                _out = nbops.mask_i_(_out, data, mask_value=0.0)

            if ipass == 0:
                data = self._update_q(data, _out, delta_q=False)
            elif ipass < _npass - 1:
                data = self._update_q(data, _out, delta_q=True)
            else:
                data["aim"] = _out

        property_data = {
            "esp": {
                "a": data["a"].clone(),
                "charges": data["charges"].clone()
            },
            "electronic": {
                "a": data["a"].clone(),
                "charges": data["charges"].clone()
            },
        }

        _extra_in_esp = torch.cat(
            [data["aim"],
             self._prepare_in_a(data),
             self._prepare_in_q(data)],
            dim=-1)

        _out = self.extra_mlps[0](_extra_in_esp)
        if data["_input_padded"].item():
            _out = nbops.mask_i_(_out, data, mask_value=0.0)

        # Update property-specific state
        _q, _f, delta_a = _out.split(
            [
                self.num_charge_channels,
                self.num_charge_channels,
                _out.shape[-1] - 2 * self.num_charge_channels,
            ],
            dim=-1,
        )
        property_data["esp"]["charges"] = property_data["esp"]["charges"] + _q
        property_data["esp"]["a"] = property_data["esp"][
            "a"] + delta_a.view_as(property_data["esp"]["a"])

        # Second MLP - read from updated property state
        _in = torch.cat([
            self._prepare_property_in_a(property_data["esp"], data),
            self._prepare_property_in_q(property_data["esp"], data)
        ],
                        dim=-1)

        _out = self.extra_mlps[1](_in)
        if data["_input_padded"].item():
            _out = nbops.mask_i_(_out, data, mask_value=0.0)

        # Update property-specific state again
        _q, _f, delta_a = _out.split(
            [
                self.num_charge_channels,
                self.num_charge_channels,
                _out.shape[-1] - 2 * self.num_charge_channels,
            ],
            dim=-1,
        )
        property_data["esp"]["charges"] = property_data["esp"]["charges"] + _q
        property_data["esp"]["a"] = property_data["esp"][
            "a"] + delta_a.view_as(property_data["esp"]["a"])

        # Third MLP - read from updated property state to generate final representation
        _in = torch.cat([
            self._prepare_property_in_a(property_data["esp"], data),
            self._prepare_property_in_q(property_data["esp"], data)
        ],
                        dim=-1)

        _extra_out_esp = self.extra_mlps[2](_in)
        if data["_input_padded"].item():
            _extra_out_esp = nbops.mask_i_(_extra_out_esp,
                                           data,
                                           mask_value=0.0)

        data["aim_enhancedesp"] = _extra_out_esp

        # Process Electronic Levels property (MLPs 3-5)
        # (Same pattern as above)
        _extra_in_electronic = torch.cat(
            [data["aim"],
             self._prepare_in_a(data),
             self._prepare_in_q(data)],
            dim=-1)

        _out = self.extra_mlps[3](_extra_in_electronic)
        if data["_input_padded"].item():
            _out = nbops.mask_i_(_out, data, mask_value=0.0)

        _q, _f, delta_a = _out.split(
            [
                self.num_charge_channels,
                self.num_charge_channels,
                _out.shape[-1] - 2 * self.num_charge_channels,
            ],
            dim=-1,
        )
        property_data["electronic"][
            "charges"] = property_data["electronic"]["charges"] + _q
        property_data["electronic"]["a"] = property_data["electronic"][
            "a"] + delta_a.view_as(property_data["electronic"]["a"])

        _in = torch.cat([
            self._prepare_property_in_a(property_data["electronic"], data),
            self._prepare_property_in_q(property_data["electronic"], data)
        ],
                        dim=-1)

        _out = self.extra_mlps[4](_in)
        if data["_input_padded"].item():
            _out = nbops.mask_i_(_out, data, mask_value=0.0)

        _q, _f, delta_a = _out.split(
            [
                self.num_charge_channels,
                self.num_charge_channels,
                _out.shape[-1] - 2 * self.num_charge_channels,
            ],
            dim=-1,
        )
        property_data["electronic"][
            "charges"] = property_data["electronic"]["charges"] + _q
        property_data["electronic"]["a"] = property_data["electronic"][
            "a"] + delta_a.view_as(property_data["electronic"]["a"])

        _in = torch.cat([
            self._prepare_property_in_a(property_data["electronic"], data),
            self._prepare_property_in_q(property_data["electronic"], data)
        ],
                        dim=-1)

        _extra_out_electronic = self.extra_mlps[5](_in)
        if data["_input_padded"].item():
            _extra_out_electronic = nbops.mask_i_(_extra_out_electronic,
                                                  data,
                                                  mask_value=0.0)

        data["aim_enhanced_electronic"] = _extra_out_electronic

        # squeeze charges
        if self.num_charge_channels == 2:
            data = self._postprocess_spin_polarized_charge(data)
        else:
            data["charges"] = data["charges"].squeeze(-1)
            data["charge"] = data["charge"].squeeze(-1)

        # readout
        for m in self.outputs.children():
            data = m(data)

        return data
