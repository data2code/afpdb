from setuptools import setup, find_packages

setup(
    name='afpdb',
    version='0.2.3',
    description='A Numpy-based PDB structure manipulation package',
    url='https://github.com/data2code/afpdb',
    author='Yingyao Zhou',
    author_email='yingyao.zhou@novartis.com',
    license= 'MIT',
    python_requires=">=3.7",
    zip_safe=False,
    package_dir={'': 'src'},
    #packages=find_packages(where='src'),
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
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
)
