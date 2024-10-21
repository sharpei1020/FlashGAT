#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include "gcn.cuh"
#include "gat.cuh"
#include "preprocess.cuh"
#include "rabbit_order/reorder.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gcn", &GCN, "gcn");
    m.def("gat", &GAT, "gat");
    m.def("preprocess_CSR", &preprocess_CSR, "preprocess_CSR");
    m.def("get_graph_set", &get_graph_set, "get_graph_set");
    m.def("process_DTC", &process_DTC, "process_DTC");
    m.def("reorder", &rabbit_reorder, "Get the reordered node id mapping: old_id --> new_id");
}
