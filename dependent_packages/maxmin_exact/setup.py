


from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='maxmin_exact',
      version="0.0.0",
      packages=find_packages(),
      install_requires=["numpy", "torch"],
      ext_modules=[cpp_extension.CppExtension('maxmin_cpp',
                                              ['./maxmin_exact/maxMin.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
