from setuptools import setup, find_packages

setup(
    name='webcam_autoencoder',
    version='0.1',
    install_requires=[
		'pygame==2.5.2',
		'numpy==1.26.4',
		'tensorflow==2.16.1',
		'keras==3.3.3',
		'matplotlib==3.9.0',
		'pillow==10.3.0',
		'opencv-python==4.10.0.82'
    ],
	entry_points={
		'console_scripts': [
			'webcam_autoencoder=main:main',
		],
	},
    python_requires='>=3.9',
)
