from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Read and return a list of requirements from the given file."""
    requirements: List[str] = []
    try:
        with open(file_path, encoding='utf-8') as file_obj:
            requirements = [line.strip() for line in file_obj if line.strip() and not line.startswith('#')]
            # remove editable install token if present
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
    except FileNotFoundError:
        # If there's no requirements.txt, return an empty list (install_requires will be empty)
        pass
    return requirements

# Read long description (README) safely
def read_readme(readme_path: str) -> str:
    try:
        with open(readme_path, encoding='utf-8') as fh:
            return fh.read()
    except FileNotFoundError:
        return ""

setup(
    name='twitter-sentiment-toxicity',
    version='0.1.0',
    author='Pankti Singh',
    author_email='panktisingh16@gmail.com',  # <-- replace with your email
    description='A reproducible machine learning pipeline for Twitter sentiment classification and toxicity detection.',
    long_description=read_readme('README.md'),
    long_description_content_type='text/markdown',
    url='hhttps://github.com/pankti7772/Twitter-Sentiment-Analysis-NLP',  # <-- replace with your repo URL
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        # Example console script: replace `twitter_toxicity.cli:main` with your actual module and function
        'console_scripts': [
            'twitter-toxicity=pipeline.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Communications :: Chat',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    keywords='nlp sentiment-analysis toxicity twitter machine-learning',
    python_requires='>=3.8',
)
