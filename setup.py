import setuptools


with open("README.md", "r") as f:
    readme = f.read()


setuptools.setup(
    name='Gil-Galad',
    version='0.1',
    author='David Reiman',
    author_email='dreiman@ucsc.edu',
    maintainer='David Reiman',
    maintainer_email='dreiman@ucsc.edu',
    description='A deep learning project template for TensorFlow',
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='deep-learning tensorflow machine-learning',
    url='https://github.com/davidreiman/Gil-Galad',
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow>=1.7.0',
        'numpy>=1.13.3',
        'parameter-sherpa',
        'tqdm',
        'progress',
    ],
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
 )
