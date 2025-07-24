from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='VETRIS',
        version='1.0.0',
        description='VETRIS: ViscoElastic Tissue-Robot Interaction Simulation with Material Point Method',
        author='Krushang Gabani',
        author_email='krushang@buffalo.edu',
        url='https://github.com/krushanggabani/VETRIS',  # Update with your GitHub URL
        keywords='Physics Simulation, Material Point Method, Soft Robotics, Viscoelastic Tissues',
        packages=find_packages(exclude=["examples", "tests", "docs"]),
        python_requires='>=3.8',
        install_requires=[
            "cuda-python",
            "gym",
            "imageio",
            "imageio-ffmpeg",
            "matplotlib",
            "numpy",
            "opencv-python",
            "open3d",
            "pandas",
            "pyglet",
            "scipy",
            "taichi",
            "torch",
            "torchvision",
            "pyyaml",
            "pyrender",
            "yacs",
            "trimesh"
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Physics",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        license="MIT",
    )
