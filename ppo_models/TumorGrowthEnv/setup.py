from setuptools import find_packages, setup

setup(name='tumor_growth',
      version='0.0.1',
      packages=find_packages(),
      package_data={
          'tumor_growth.envs': [
              'data/default-parameters.json',
              'data/tumor-lib/*.txt',
          ],
      },
      include_package_data=True,
      install_requires=['gym', 'numpy', 'emt6ro']
      )
