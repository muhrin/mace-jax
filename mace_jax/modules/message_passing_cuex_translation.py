import math
from typing import Callable

import cuequivariance as cue
import cuequivariance_jax as cuex
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import haiku as hk


from . import convert

class MessagePassingConvolution(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        target_irreps: e3nn.Irreps,
        max_ell: int,
        activation: Callable,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.max_ell = max_ell
        self.activation = activation

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2

        # Translate everything to cuex
        node_feats = convert.e3j2cuex_irreps_array(node_feats)
        sph = convert.e3j2cuex_irreps_array(
            e3nn.spherical_harmonics(range(0, self.max_ell + 1), vectors, True)
        )
        target_irreps = convert.e3j2cuex_irreps(self.target_irreps)
        dtype = node_feats.dtype

        descriptor = cue.descriptors.channelwise_tensor_product(
            node_feats.irreps, sph.irreps, target_irreps
        )
        descriptor = 1.0 / math.sqrt(self.avg_num_neighbors) * descriptor

        w = e3nn.haiku.MultiLayerPerceptron(
            3 * [64] + [descriptor.inputs[0].dim],
            self.activation,
            output_activation=False,
            with_bias=False,
        )(
            radial_embedding
        )  # [n_edges, num_irreps]

        node_feats = cuex.equivariant_polynomial(
            descriptor,
            [w.array, node_feats, sph],
            outputs_shape_dtype=jax.ShapeDtypeStruct((node_feats.shape[0], -1), dtype),
            indices=[None, senders, None, receivers],
            name=f"{self.name}_TP",
        )

        return convert.cuex_to_e3j_irreps_array(node_feats)
