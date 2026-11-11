import argparse

from alembic import command
from alembic.config import Config

from .config import ROOT


def main():
    parser = argparse.ArgumentParser(description="DataClean 数据库管理")
    parser.add_argument("action", choices=["init-db"])
    parser.parse_args()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    print("数据库迁移完成")


if __name__ == "__main__":
    main()
