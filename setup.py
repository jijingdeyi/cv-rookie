# setup.py  —— 放在仓库根目录
from pathlib import Path

from setuptools import find_packages, setup

# --------- 基本信息 ---------
NAME = "cv_rookie"                # pip install 后的 import 名字
DESCRIPTION = "Computer-Vision utils and models for rookies"
URL = "https://github.com/jijingdeyi/cv-rookie"
EMAIL = "jijingdeyi@gmail.com"  # 可留空
AUTHOR = "jijingdeyi"
LICENSE = "MIT"                   # 或仓库里实际的 licence
PYTHON_REQUIRES = ">=3.7"

# --------- 版本号策略 ---------
# 你可以手动写，也可以读取包里 __init__.__version__
# 从包的 __init__.py 里抓 __version__
VERSION = '0.1.0'

# --------- 依赖 ---------
# 如果仓库已经有 requirements.txt，读进去就行
def parse_reqs(path="requirements.txt"):
    req_file = Path(__file__).with_name(path)
    if req_file.exists():
        return [
            line.strip()
            for line in req_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []

INSTALL_REQUIRES = parse_reqs()

# 可选：开发 / 测试 额外依赖
EXTRAS_REQUIRE = {
    "dev": ["black", "isort", "pytest", "pre-commit"],
}

# --------- 真·setup 调用 ---------
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=(Path(__file__).with_name("README.md").read_text()
                      if Path(__file__).with_name("README.md").exists()
                      else DESCRIPTION),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # 如果有 package_data / MANIFEST.in
    zip_safe=False,
)
