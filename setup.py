from setuptools import setup, find_packages

setup(
    name="powerx",
    version="1.0.0",
    description="PowerX: Pool idle local accelerators and cloud GPUs into a unified AI compute mesh with Unified Memory & NVIDIA DCGM Telemetry",
    packages=find_packages(),
    package_data={"hybrid_gpu_mesh": ["dashboard.html"]},
    python_requires=">=3.9",
    install_requires=[
        "psutil>=5.9.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "test": ["pytest>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "powerx=hybrid_gpu_mesh.cli:main",
            "hybrid-gpu-mesh=hybrid_gpu_mesh.cli:main",
        ],
    },
)
