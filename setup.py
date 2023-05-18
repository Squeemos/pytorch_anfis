from setuptools import setup

def main() -> int:
    packages = ["anfis"]
    setup(name="anfis", version="1.0", packages=packages)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
