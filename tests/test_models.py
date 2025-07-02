from collections import namedtuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from e3nn_jax.utils import assert_equivariant

from mace_jax import modules
from mace_jax.modules import MACE, SymmetricContraction


def test_symmetric_contraction():
    x = e3nn.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    y = jax.random.normal(jax.random.PRNGKey(1), (32, 4))

    model = hk.without_apply_rng(
        hk.transform(lambda x, y: SymmetricContraction(3, ["0e", "1o", "2e"])(x, y))
    )
    w = model.init(jax.random.PRNGKey(2), x, y)

    assert_equivariant(
        lambda x: model.apply(w, x, y), jax.random.PRNGKey(3), args_in=(x,)
    )


# TODO fix this test
def test_mace():
    num_species = 2

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

    w = model.init(jax.random.PRNGKey(0), graph)

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
        energy = model.apply(w, graph)["energy"]
        return e3nn.IrrepsArray("0e", energy)

    positions = e3nn.normal("1o", jax.random.PRNGKey(1), (2,))
    assert_equivariant(wrapper, jax.random.PRNGKey(1), positions)


if __name__ == "__main__":
    test_mace()
    # test_symmetric_contraction()
