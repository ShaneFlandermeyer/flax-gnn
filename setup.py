from setuptools import setup, find_packages

setup(
    name='flax-gnn',
    version='0.1.0',    
    description='Graph neural networks with Jraph + Flax',
    url='https://github.com/ShaneFlandermeyer/flax-gnn',
    author='Shane Flandermeyer',
    author_email='shaneflandermeyer@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'jax',
      'jaxlib',
      'flax',
      'numpy',
      'optax',
      'jaxtyping',
      'einops',
      'jraph',
    ],

)