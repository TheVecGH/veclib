from setuptools import setup, find_packages

setup(
    name='veclib',
    version='0.1',
    description='A library for symbolic tensor calculus in general relativity.',
    author='Niels Slotboom',
    packages=find_packages(),
    install_requires=[
        'sympy',
        # Add other dependencies here
    ],
    python_requires='>=3.7',  # Adjust according to your Python version
)
