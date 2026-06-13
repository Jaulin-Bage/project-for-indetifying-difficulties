import argparse
import csv
import os
import shutil
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULT_OPTIMIZED = Path("true_data/hph/optimized_indicators.csv")
DEFAULT_TARGET = Path("tmp/整合版量化指标体系_20260613.csv")


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def read_csv(path):
    last_error = None

    for encoding in ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return reader.fieldnames, rows, encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"无法用这些编码读取 {path}: {', '.join(ENCODINGS)}",
    ) from last_error


def write_csv(path, fieldnames, rows):
    tmp_path = path.with_name(path.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, path)


def clean_name(value):
    return (value or "").strip()


def format_decimal(value):
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def load_weight_map(optimized_path):
    _, rows, _ = read_csv(optimized_path)
    weight_map = {}

    for row_no, row in enumerate(rows, start=2):
        name = clean_name(row.get("指标名称"))
        raw_weight = clean_name(row.get("优化后权重"))

        if not name:
            continue
        if not raw_weight:
            raise ValueError(f"{optimized_path}:{row_no} 缺少优化后权重")

        try:
            weight_map[name] = Decimal(raw_weight) * Decimal("100")
        except InvalidOperation as exc:
            raise ValueError(
                f"{optimized_path}:{row_no} 优化后权重不是有效数字: {raw_weight}"
            ) from exc

    return weight_map


def update_score_max(optimized_path, target_path, output_path, backup, dry_run):
    weight_map = load_weight_map(optimized_path)
    fieldnames, rows, _ = read_csv(target_path)

    required_columns = {"indicator_2", "score_max"}
    missing_columns = required_columns - set(fieldnames or [])
    if missing_columns:
        raise ValueError(f"{target_path} 缺少列: {', '.join(sorted(missing_columns))}")

    matched_indicators = set()
    changed_rows = 0

    for row in rows:
        indicator_name = clean_name(row.get("indicator_2"))
        if indicator_name not in weight_map:
            continue

        new_score_max = format_decimal(weight_map[indicator_name])
        if row.get("score_max") != new_score_max:
            row["score_max"] = new_score_max
            changed_rows += 1
        matched_indicators.add(indicator_name)

    unmatched_optimized = sorted(set(weight_map) - matched_indicators)
    unmatched_target = sorted(
        {
            clean_name(row.get("indicator_2"))
            for row in rows
            if clean_name(row.get("indicator_2")) and clean_name(row.get("indicator_2")) not in weight_map
        }
    )

    print(f"优化表指标数: {len(weight_map)}")
    print(f"目标表行数: {len(rows)}")
    print(f"匹配到的指标数: {len(matched_indicators)}")
    print(f"更新 score_max 的行数: {changed_rows}")

    if unmatched_optimized:
        print("\n优化表中未匹配到目标表的指标:")
        for name in unmatched_optimized:
            print(f"  - {name}")

    if unmatched_target:
        print("\n目标表中未匹配到优化表的 indicator_2:")
        for name in unmatched_target:
            print(f"  - {name}")

    if dry_run:
        print("\n已执行预览，未写入文件。")
        return

    if output_path == target_path and backup:
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
        print(f"\n已备份原文件: {backup_path}")

    write_csv(output_path, fieldnames, rows)
    print(f"已写入: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 optimized_indicators.csv 的优化后权重 * 100 写入量化指标体系 score_max 列。"
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        default=DEFAULT_OPTIMIZED,
        help=f"优化后指标权重 CSV，默认: {DEFAULT_OPTIMIZED}",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"待更新的量化指标体系 CSV，默认: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径。默认覆盖 target，并自动生成 .bak 备份。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="覆盖 target 时不生成 .bak 备份。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示匹配和更新统计，不写入文件。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    optimized_path = args.optimized
    target_path = args.target
    output_path = args.output or target_path

    update_score_max(
        optimized_path=optimized_path,
        target_path=target_path,
        output_path=output_path,
        backup=args.no_backup,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
