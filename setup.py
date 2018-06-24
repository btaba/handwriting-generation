from distutils.core import setup
from setuptools import find_packages


def setup_package():
    config = {
        'name': 'handwriting-generation',
        'version': '0.0.1',
        'description': 'handwriting generation',
        'author': 'Baruch Tabanpour',
        'author_email': 'baruch@tabanpour.info',
        'url': 'https://github.com/btaba/handwriting-generation',
        'license': 'MIT',
        'tests_require': ['pytest'],
        'packages': find_packages(
            exclude=("tests", )),
        'keywords': [
            'handwriting generation',
            'scribe'
        ]
    }

    setup(**config)


if __name__ == '__main__':
    setup_package()
