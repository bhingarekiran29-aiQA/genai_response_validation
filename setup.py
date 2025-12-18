from setuptools import setup, find_packages

setup(
    name="genai_response_validation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "python-dotenv",
        "deepeval"
    ],
)