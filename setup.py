from setuptools import setup

setup(
    name='madpy-seis',
    version='0.1.0',    
    description='Measure Amplitude and Duration in Python',
    url='https://github.com/seismomomo/madpy',
    author='Monique Holt',
    author_email='mmholt@uic.edu',
    license='MIT',
    packages=['madpy', 'madpy.plotting', 'madpy.tests', 'madpy.tests.testdata'],
    install_requires=['jupyterlab==3.2.1',
                      'matplotlib==3.4.3',
                      'obspy==1.2.2',
                      'pandas==1.3.4',
                      'threadpoolctl==3.0.0'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
    ],
)
