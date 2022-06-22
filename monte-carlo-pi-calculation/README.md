# PI estimation implementations

Example programs to estimate pi value using Monte Carlo method

# Building
SW requires Poplar SDK to be set up.
`$ make`

# Running

```
./cpu_ip --iterations XXXXX
./all_ipu_pi --iterations XXXXX --chunk_size 10000000 [--num_ipus xxx]
./vertex_ipu_pi --iterations XXXXX [--num_ipus xxx]
```

# Implementations

Every implementation generate random point and check it against x^2 + y^2 <= 1 and increments count `--iterations` times.

## CPU pi

CPU openMP implementation. Takes only --iterations argument, rest is ignored.

## Poplar

### Poplibs implementatio (all_ipu_pi)

Implementation creates `--chunk_size` buffer and fills it from IPU.
After computation is done everything is copied to host.

### MultiVertex implementation

Implementations creates output tensor with `num_ipus * tiles * threads` elements.

### Generated implementation (obsolete)

Left for reference. It generates random data on host, process it on IPU and copy it back to host.

### Iterative implementation (obsolete)

Initial implementation taking huge tensors with host generated random data and returning comparison results.
