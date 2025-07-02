import e3nn_jax as e3nn
import cuequivariance as cue
import cuequivariance_jax as cuex


def e3j2cuex_irreps(ir: e3nn.Irreps) -> cue.Irreps:
    return cue.Irreps(cue.O3, ir)


def e3j2cuex_irreps_array(ira: e3nn.IrrepsArray) -> cuex.RepArray:
    """Convert e3nn-jax IrrepsArray to a cuex.RepArray"""
    ira = cuex.RepArray(e3j2cuex_irreps(ira.irreps), ira.array, cue.mul_ir)
    ira = ira.change_layout(cue.ir_mul)
    return ira


def cuex_to_e3j_irreps_array(rep_array: cuex.RepArray) -> e3nn.IrrepsArray:
    z = rep_array.change_layout(cue.mul_ir)
    z = e3nn.IrrepsArray(str(rep_array.irreps), z.array)
    return z
