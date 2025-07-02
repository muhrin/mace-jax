"""
Implementation from MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields
Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner and Gábor Csányi
"""

from typing import Any, Callable, Optional, Set, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn
from cuequivariance.group_theory.experimental.mace import symmetric_contractions
import cuequivariance_jax as cuex

from . import convert


class SymmetricContraction(hk.Module):
    r"""Symmetric tensor product contraction with parameters

    Equivalent to the following code executed in parallel on the channel dimension::

        e3nn.haiku.Linear(irreps_out)(
            e3nn.concatenate([
                x,
                tensor_product(x, x),  # additionally keeping only the symmetric terms
                tensor_product(tensor_product(x, x), x),
                ...
            ])
        )

    Each channel has its own parameters.

    Args:
        orders (tuple of int): orders of the tensor product
        keep_irrep_out (optional, set of Irrep): irreps to keep in the output
        get_parameter (optional, callable): function to get the parameters, by default it uses ``hk.get_parameter``
            it should have the signature ``get_parameter(name, shape) -> Array`` and return a normal distribution
            with variance 1
    """

    def __init__(
            self,
            correlation: int,
            keep_irrep_out: Set[e3nn.Irrep],
            num_species: int,
            gradient_normalization: str | float = None,
            symmetric_tensor_product_basis: bool = True,
            off_diagonal: bool = False,
    ):
        super().__init__()
        self.correlation = correlation

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]
        self.gradient_normalization = gradient_normalization

        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        self.keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal

    def __call__(self, x: e3nn.IrrepsArray, index: jnp.ndarray) -> e3nn.IrrepsArray:
        r"""Evaluate the symmetric tensor product

        Args:
            x (IrrepsArray): input of shape ``(..., num_channel, irreps)``

        Returns:
            IrrepsArray: output of shape ``(..., num_channel, irreps_out)``
        """

        # TODO: normalize by taking into account the correlation, like in TensorSquare
        # TODO: what do we do with num_channel?
        # - This operation is parallel on the feature dimension (but each feature has its own parameters)
        assert x.ndim == 3  # [num_nodes, num_channel, irreps_x.dim]

        out = set()

        weights = []

        for order in range(self.correlation, 0, -1):  # correlation, ..., 1
            U = e3nn.reduced_tensor_product_basis(
                [x.irreps] * order, keep_ir=self.keep_irrep_out
            )
            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            for (mul, ir_out), u in zip(U.irreps, U.list):
                w = hk.get_parameter(
                    f"w{order}_{ir_out}",
                    (self.num_species, mul, x.shape[1]),
                    dtype=jnp.float32,
                    init=hk.initializers.RandomNormal(
                        stddev=(mul ** -0.5) ** (1.0 - self.gradient_normalization)
                    ),
                )
                print(mul, ir_out, x.shape[1])

                weights.append(w)
                # w = (
                #         w * (mul ** -0.5) ** self.gradient_normalization
                # )  # normalize weights

                out.add(ir_out)

        irreps_out = e3nn.Irreps(sorted(out))

        mul = x.shape[1]
        x = x.axis_to_mul()

        descriptor, proj = symmetric_contractions.symmetric_contraction(
            convert.e3j2cuex_irreps(x.irreps),
            mul * convert.e3j2cuex_irreps(irreps_out),
            tuple(range(1, self.correlation + 1)),
        )

        weights = jnp.concatenate(weights, axis=1)  # num_species, <something>, mul

        w = jnp.einsum("zau,ab->zbu", weights, proj)
        w = jnp.reshape(w, (self.num_species, -1))

        y = cuex.equivariant_polynomial(
            descriptor, [w, convert.e3j2cuex_irreps_array(x)],
            indices=[index, None, None]
        )

        y = convert.cuex_to_e3j_irreps_array(y)
        y = y.mul_to_axis()

        return y
