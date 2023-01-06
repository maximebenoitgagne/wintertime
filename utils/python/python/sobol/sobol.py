from .sobol_lib import i4_bit_hi1, i4_bit_lo0, i4_sobol_generate, i4_sobol, i4_uniform, prime_ge, isprime
from .sobol_lib2 import r4_uniform_01, r8mat_write, tau_sobol

def sobol(dim_num, seed):
    while True:
        v, seed = i4_sobol(dim_num, seed)
        yield v
