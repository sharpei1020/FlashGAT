#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include "gcn.cuh"
#include "gat.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gcn", &gcn, "gcn");
    m.def("gat", &gat, "gat");
}
