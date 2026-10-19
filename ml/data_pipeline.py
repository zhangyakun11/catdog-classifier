"""数据流水线：从 Dogs vs Cats 原始数据生成两个数据集版本。

- v1: 每类随机抽样 N 张原始图片 -> data/raw/  （第一版可用数据）
- v2: 清洗（丢弃损坏/非JPEG图片）+ 统一缩放到 IMG_SIZE -> data/processed/（改进版数据）

用法：
    python ml/data_pipeline.py --source <PetImages目录> --n 200 --size 128

输出：
    data/raw/          -> v1（由 DVC 版本管理）
    data/processed/    -> v2（由 DVC 版本管理）
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def sample_and_copy(source_dir: Path, dest_dir: Path, n_per_class: int, seed: int = 42):
    """v1: 从 source 每类随机抽 n 张，原样复制到 dest_dir。"""
    random.seed(seed)
    total = 0
    for cls in ["Cat", "Dog"]:
        src = source_dir / cls
        images = sorted(src.glob("*.jpg"))
        chosen = random.sample(images, min(n_per_class, len(images)))
        out = dest_dir / cls.lower()
        out.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(chosen):
            shutil.copy2(img, out / f"{cls.lower()}_{i:04d}.jpg")
        total += len(chosen)
        print(f"  v1 [{cls}] -> 抽取 {len(chosen)} 张 -> {out}")
    print(f"  v1 共 {total} 张（原始大小，未清洗）")


def clean_and_resize(src_dir: Path, dest_dir: Path, size: int):
    """v2: 验证图片完整性（丢弃损坏/非JPEG），统一缩放为 size x size。"""
    kept, dropped = 0, 0
    drop_reasons = {}
    for cls in ["cat", "dog"]:
        src = src_dir / cls
        out = dest_dir / cls
        out.mkdir(parents=True, exist_ok=True)
        for img in sorted(src.glob("*.jpg")):
            try:
                with Image.open(img) as im:
                    im.verify()  # 验证文件头/结构
                with Image.open(img) as im:
                    im = im.convert("RGB")
                    im = im.resize((size, size), Image.LANCZOS)
                    im.save(out / img.name, "JPEG", quality=90)
                kept += 1
            except (UnidentifiedImageError, OSError, ValueError) as e:
                dropped += 1
                reason = type(e).__name__
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
    print(f"  v2 保留 {kept} 张，丢弃损坏 {dropped} 张（原因分布: {drop_reasons}）")
    print(f"  v2 输出 -> {dest_dir}（{size}x{size} 统一尺寸）")


def main():
    parser = argparse.ArgumentParser(description="猫狗数据集数据流水线")
    parser.add_argument("--source", type=str, required=True,
                        help="PetImages 目录（含 Cat/ 和 Dog/ 子目录）")
    parser.add_argument("--n", type=int, default=200, help="每类抽样张数（v1）")
    parser.add_argument("--size", type=int, default=128, help="v2 缩放尺寸")
    args = parser.parse_args()

    source = Path(args.source)
    assert (source / "Cat").is_dir() and (source / "Dog").is_dir(), \
        f"source 必须包含 Cat/ 和 Dog/ 子目录: {source}"

    print("=== 生成数据集 v1（原始抽样）===")
    raw_dir = DATA_DIR / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    sample_and_copy(source, raw_dir, args.n)

    print("=== 生成数据集 v2（清洗 + 缩放）===")
    proc_dir = DATA_DIR / "processed"
    if proc_dir.exists():
        shutil.rmtree(proc_dir)
    clean_and_resize(raw_dir, proc_dir, args.size)

    print("=== 完成 ===")
    print(f"v1 -> {raw_dir}")
    print(f"v2 -> {proc_dir}")


if __name__ == "__main__":
    main()
