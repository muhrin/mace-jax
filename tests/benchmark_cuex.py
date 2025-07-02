from collections import namedtuple
import os
from typing import Final
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jraph
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

import cuequivariance.group_theory.experimental.mace.symmetric_contractions

from mace_jax.modules import MACE

num_species = 2
IMPLEMENTATION: Final[str] = os.environ["IMPLEMENTATION"]
rng_key = jax.random.PRNGKey(0)


@hk.without_apply_rng
@hk.transform
def model(graph):
    positions = graph.nodes.positions
    receivers = graph.receivers
    senders = graph.senders
    receivers_unit_shifts = graph.edges.shifts
    vectors = (positions[receivers] + receivers_unit_shifts) - positions[senders]
    species = graph.nodes.species

    return MACE(
        r_max=5,
        radial_basis=lambda r, r_max: e3nn.bessel(r, 8, r_max),
        radial_envelope=lambda r, r_max: e3nn.poly_envelope(5 - 1, 2, r_max)(r),
        max_ell=2,
        num_interactions=5,
        num_species=num_species,
        hidden_irreps=e3nn.Irreps("32x0e"),
        readout_mlp_irreps=e3nn.Irreps("16x0e"),
        gate=jax.nn.silu,
        avg_num_neighbors=8,
        correlation=3,
        output_irreps="0e",
        implementation=IMPLEMENTATION,
    )(vectors, species, senders, receivers)


Node = namedtuple("Node", ["positions", "species", "attrs"])
Edge = namedtuple("Edge", ["shifts"])
Globals = namedtuple("Globals", ["cell"])

graph = jraph.GraphsTuple(
    nodes=Node(
        positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        species=jnp.array([0, 1]),
        attrs=jax.nn.one_hot(jnp.array([0, 1]), num_species),
    ),
    edges=Edge(shifts=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    globals=Globals(cell=None),
    senders=jnp.array([0, 1]),
    receivers=jnp.array([1, 0]),
    n_edge=jnp.array([2]),
    n_node=jnp.array([2]),
)

params = model.init(rng_key, graph)


def wrapper(positions):
    species = jnp.array([0, 1])
    graph = jraph.GraphsTuple(
        nodes=Node(
            positions=positions.array,
            species=species,
            attrs=jax.nn.one_hot(species, num_species),
        ),
        edges=Edge(shifts=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        globals=Globals(cell=None),
        senders=jnp.array([0, 1]),
        receivers=jnp.array([1, 0]),
        n_edge=jnp.array([2]),
        n_node=jnp.array([2]),
    )
    detailed = model.apply(params, graph)

    print(detailed.array)
    energy = detailed.array.sum()
    return e3nn.IrrepsArray("0e", energy)


positions = e3nn.normal("1o", rng_key, (2,))


t0 = time.time()
for _ in range(10):
    res = wrapper(positions)

jax.block_until_ready(res)
t1 = time.time()

print("Delta", t1 - t0)
print(res)
