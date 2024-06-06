from setuptools import setup
from setuptools import find_packages

setup(
    name='afpdb',
    version='0.2.0',
    description='A numpy-based PDB structure manipulation package',
    url='https://github.com/data2code/afpdb',
    author='Yingyao Zhou',
    author_email='yingyao.zhou@novartis.com',
    license='Apache-2.0',
    zip_safe=False,
    packages=["afpdb","afpdb.mycolabfold","afpdb.myalphafold","afpdb.myalphafold.common","afpdb.mycolabdesign"],
    install_requires=['pandas',
                      'numpy',
                      'scipy',
                      'biopython',
                      'dm-tree',
                      'py3Dmol',
                      'tabulate',
                      'requests'
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
)
