import argparse
import shutil
from pathlib import Path


DEFAULT_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
)


def read_text_with_encoding(path, from_encoding=None):
    encodings = (from_encoding,) if from_encoding else DEFAULT_ENCODINGS
    last_error = None

    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            return text, encoding
        except UnicodeError as exc:
            last_error = exc

    raise UnicodeError(
        f"无法识别文件编码: {path}，已尝试: {', '.join(encodings)}"
    ) from last_error


def convert_one_file(path, output_encoding, from_encoding, backup, dry_run):
    text, detected_encoding = read_text_with_encoding(path, from_encoding)

    print(f"{path}: {detected_encoding} -> {output_encoding}")
    if dry_run:
        return

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"  备份: {backup_path}")
        else:
            print(f"  备份已存在，跳过: {backup_path}")

    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding=output_encoding, newline="")
    tmp_path.replace(path)


def collect_files(paths, recursive, suffixes):
    files = []
    suffixes = {suffix.lower() for suffix in suffixes}

    for path in paths:
        if path.is_file():
            files.append(path)
            continue

        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in path.glob(pattern):
                if child.is_file() and child.suffix.lower() in suffixes:
                    files.append(child)
            continue

        raise FileNotFoundError(f"路径不存在: {path}")

    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="把 CSV/TXT 等文本文件统一转换为 UTF-8-SIG 编码。"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="要转换的文件或目录，例如 true_data/total.csv",
    )
    parser.add_argument(
        "--to-encoding",
        default="utf-8-sig",
        help="目标编码，默认 utf-8-sig。若不需要 BOM，可改为 utf-8。",
    )
    parser.add_argument(
        "--from-encoding",
        default=None,
        help="强制指定源编码，例如 gb18030。不填则自动尝试常见编码。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="路径是目录时，递归转换子目录文件。",
    )
    parser.add_argument(
        "--suffix",
        nargs="+",
        default=[".csv", ".txt"],
        help="目录模式下要转换的后缀，默认 .csv .txt。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不生成 .bak 备份。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示会转换哪些文件，不写入。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = collect_files(args.paths, args.recursive, args.suffix)

    if not files:
        print("没有找到需要转换的文件。")
        return

    for path in files:
        convert_one_file(
            path=path,
            output_encoding=args.to_encoding,
            from_encoding=args.from_encoding,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
