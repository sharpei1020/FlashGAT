#include <torch/serialize/tensor.h>
#include <torch/extension.h>
// #include "gcn.cuh"
#include "gat.cuh"
#include "agnn.cuh"
#include "e2v_gat_kernel.cuh"
#include "preprocess.cuh"
#include "rabbit_order/reorder.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("e2v_gat", &E2V_GAT, "e2v_gat");
    m.def("agnn", &AGNN, "agnn");
    m.def("agnn_udf", &AGNN_UDF, "agnn_udf");
    m.def("agnn_short", &AGNN_short, "agnn_short");
    m.def("agnn_adaptive", &AGNN_adaptive, "agnn_adaptive");
    m.def("SDDMM", &SDDMM, "SDDMM");
    m.def("SDDMM_TCGNN", &SDDMM_TCGNN, "SDDMM_TCGNN");
    m.def("agnn_divide", &AGNN_divide, "AGNN_divide");
    // m.def("gcn", &GCN, "gcn");
    // m.def("gat", &GAT, "gat");
    m.def("gat_balance", &GAT_balance, "gat_balance");
    m.def("gat_short", &GAT_short, "gat_short");
    m.def("gat_adaptive", &GAT_adaptive, "gat_adaptive");
    m.def("sputnik_gat", &sputnik_GAT, "sputnik_gat");
    m.def("sputnik_agnn", &sputnik_AGNN, "sputnik_agnn");
    m.def("preprocess_CSR", &preprocess_CSR, "preprocess_CSR(with counts)");
    m.def("process_CSR", &process_CSR, "process_CSR(without counts)");
    m.def("get_graph_set", &get_graph_set, "get_graph_set");
    m.def("process_DTC", &process_DTC, "process_DTC");
    m.def("process_DTC_short_mask", &process_DTC_short_mask, "process_DTC_short_mask");
    m.def("SGT_short_Mask", &SGT_short_Mask, "SGT_short_Mask");
    m.def("DTC_compression", &DTC_compression, "DTC_compression");
    m.def("ASC_e2v", &ASC_e2v, "ASC_e2v");
    m.def("adaptive_ASC", &adaptive_ASC, "adaptive_ASC");
    // m.def("reorder", &rabbit_reorder, "Get the reordered node id mapping: old_id --> new_id");
}
