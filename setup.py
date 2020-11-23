import os
import setuptools
from distutils.command.build_ext import build_ext as DistUtilsBuildExt
from importlib.machinery import SourceFileLoader
from setuptools import Extension, find_packages, setup

import numpy

version = SourceFileLoader('cral.version',
                           os.path.join('cral',
                                        'version.py')).load_module().VERSION

extensions = [
    Extension('cral.models.object_detection.retinanet.compute_overlap',
              ['cral/models/object_detection/retinanet/compute_overlap.pyx'])
]


class BuildExtension(setuptools.Command):
    description = DistUtilsBuildExt.description
    user_options = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as \
            SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


setup(
    name='cral',
    version=version,
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=[
        'tqdm',
        'xxhash',
        'pandas',
        # 'opencv-python==3.4.2.17',
        'albumentations==0.4.5',
        'jsonpickle',
        'pycocotools',
        'pydensecrf'
    ],
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': BuildExtension},
    author='T Pratik',
    author_email='pratik@segmind.com',
    keywords=[
        'CNN', 'Deep Learning', 'classification', 'object detection',
        'segmentation', 'keras', 'tensorflow-keras'
    ],
    description='CRAL: Library for CNNs',
    long_description=open('README.md').read(),
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.6',
    ],
)
# sudo python3 setup.py build_ext --inplace
