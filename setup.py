from setuptools import find_packages, setup
import os

REQUIRES = []

try:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            REQUIRES.append(line.split("#")[0].strip())  # Remove inline comments
except FileNotFoundError:
    print("[setup.py] WARNING: requirements.txt not found. No dependencies will be installed.")
except Exception as e:
    print(f"[setup.py] ERROR reading requirements.txt: {e}")

setup(
    name="finrl-meta",
    version="0.3.7",
    author="AI4Finance Foundation",
    author_email="contact@ai4finance.org",
    url="https://github.com/AI4Finance-Foundation/FinRL-Meta",
    license="MIT",
    install_requires=REQUIRES,
    description="FinRL­-Meta: A Universe of ­Market Environments for Data­-Driven Financial Reinforcement Learning",
    packages=find_packages(),
    long_description="FinRL­-Meta: A Universe of Near Real­ Market Environments for Data­-Driven Financial Reinforcement Learning",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcement Learning",
    platforms=["any"],
    python_requires=">=3.6",
)
