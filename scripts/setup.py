from setuptools import setup, find_packages

setup(name='dined',
      version='0.1',
      description='The name of your package',
      url='',
      author='',
      author_email='',
      license='',
      packages=find_packages(),
      install_requires=[
            # Put here what is required for your package to run
      ],
      extras_require={
      # Put here any development dependencies for your package
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
            'jupyter',
            'notebook',
            'pylint'
        ]
      },
      zip_safe=False
)

