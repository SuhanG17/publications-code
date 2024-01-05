from setuptools import setup

exec(compile(open('beowulf/version.py').read(),
             'beowulf/version.py', 'exec'))

# setup(name='beowulf',
#       version=__version__,
#       description='Beowulf is the various data generating mechanisms',
#       keywords='DGM',
#       packages=['beowulf',
#                 'beowulf.dgm'],
#       include_package_data=True,
#       author='Paul Zivich'
#       )

# modified to include .csv files
setup(name='beowulf',
      version=__version__,
      description='Beowulf is the various data generating mechanisms',
      keywords='DGM',
      packages=['beowulf',
                'beowulf.dgm',
                'beowulf.data_files'], # include data_files folder
      include_package_data=True,
      package_data = {'': ['*.csv'],}, # If any package contains *.csv files, include them
      author='Paul Zivich'
      )
