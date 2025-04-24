from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


# setup(
#     name="graphiler",
#     version="0.0.1",
#     url="https://github.com/xiezhq-hermann/graphiler",
#     package_dir={'': 'python'},
#     packages=['graphiler', 'graphiler.utils'],
#     ext_modules=[
#         CppExtension('graphiler.mpdfg', [
#             'src/pybind.cpp',
#             'src/builder/builder.cpp',
#             'src/optimizer/dedup.cpp',
#             'src/optimizer/split.cpp',
#             'src/optimizer/reorder.cpp',
#             'src/optimizer/fusion.cpp'
#         ]),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     },
#     python_requires=">=3.7"
# )

setup(
    name="mygraph",
    package_dir={'':'python'},
    packages=['minhash_order'],
    ext_modules=[
        CUDAExtension('mygraph', [
            # 'src/ops/my_kernel/rabbit_order/reorder.cpp',
            'src/ops/my_kernel/bind.cc',
            'src/ops/my_kernel/preprocess.cu',
            'src/ops/my_kernel/gcn_kernel.cu',
            'src/ops/my_kernel/gat_kernel.cu',
            'src/ops/my_kernel/gat_kernel_6.cu',
            'src/ops/my_kernel/gat_kernel_adaptive.cu',
            'src/ops/my_kernel/agnn_kernel.cu',
            'src/ops/my_kernel/agnn_kernel_adaptive.cu',
            'src/ops/my_kernel/agnn_udf.cu',
            'src/ops/my_kernel/agnn_kernel_.cu',
            'src/ops/my_kernel/agnn_divide.cu',
            'src/ops/my_kernel/e2v_gat_kernel.cu',
            'src/ops/my_kernel/sputnik_gat_kernel.cu',
            'src/ops/my_kernel/sputnik_agnn_kernel.cu'
        ],
        extra_compile_args={'cxx': ['-g', '-fopenmp', '-mcx16', '-fconcepts'], 'nvcc': ['-O2', '--extended-lambda']},
        library_dirs=['/home/ljq/mine/graphiler/src/build/sputnik'],
        libraries=['numa', 'sputnik'],
        include_dirs=['/home/ljq/mine/graphiler/src'])
    ],
    cmdclass={'build_ext': BuildExtension},
)
