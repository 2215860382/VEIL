"""把老库格式重组成新库的两层结构 + 改字段名。

老库格式：
    {video_id}.json                     # bank（含 chunks[] 数组，含 memory_text）
    {video_id}/episodic/{chunk_id}.json # 每个 chunk 一个 json（含 episodic_descs）
    {video_id}/keyframes/{chunk_id}.jpg # 关键帧

新库格式：
    {video_id}/
    ├── narrative.json   # 叙述层（动态层）：从老库提取
    │   {
    │     "video_id": "...",
    │     "duration": ...,
    │     "chunks": [{
    │       "chunk_id": 0,
    │       "start_time": ..., "end_time": ...,
    │       "narrative": "...",                # 老库 memory_text
    │       "caption": ["...", ...],           # 老库 episodic_descs
    │       "frame_timestamps": [...],
    │       "speech_text": "...",
    │       "keyframe_path": "keyframes/0000.jpg"
    │     }, ...]
    │   }
    ├── attributes.json  # 属性层（静态层）：待 VLM 生成
    ├── metadata.json    # 元数据：模型、参数等
    └── keyframes/       # 关键帧（保留不动）

用法：
    PYTHONPATH=. python src/build_memory/reorganize_structure.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


BANK_DIR = Path("/home2/ycj/Project/VEIL/outputs/memory/videomme_L_27B")
PARQUET_PATH = "/home2/ycj/Datas/VideoMME/videomme/test-00000-of-00001.parquet"


def load_target_video_ids() -> list[str]:
    df = pd.read_parquet(PARQUET_PATH)
    df_long = df[df["duration"] == "long"]
    return sorted(df_long["videoID"].unique().tolist())


def reorganize_one(video_id: str) -> tuple[bool, str]:
    video_dir = BANK_DIR / video_id
    main_json = BANK_DIR / f"{video_id}.json"

    if not main_json.exists() or not video_dir.exists():
        return False, "missing source"

    # Load main bank (含 memory_text)
    with open(main_json) as f:
        bank = json.load(f)

    chunks_meta = bank.get("chunks", [])
    duration = bank.get("duration", 0.0)

    # Build narrative.json from main bank + episodic descs
    episodic_dir = video_dir / "episodic"
    new_chunks = []
    for chunk in chunks_meta:
        chunk_id = chunk["chunk_id"]
        ep_path = episodic_dir / f"{chunk_id:04d}.json"

        # Load caption from episodic file
        caption = []
        frame_timestamps = []
        speech_text = chunk.get("asr", "")
        if ep_path.exists():
            with open(ep_path) as f:
                ep = json.load(f)
            caption = ep.get("episodic_descs", [])
            frame_timestamps = ep.get("frame_timestamps", [])
            speech_text = ep.get("speech_text", speech_text)

        new_chunks.append({
            "chunk_id": chunk_id,
            "start_time": chunk.get("start_time", 0.0),
            "end_time": chunk.get("end_time", 0.0),
            "narrative": chunk.get("memory_text", ""),
            "caption": caption,
            "frame_timestamps": frame_timestamps,
            "speech_text": speech_text,
            "keyframe_path": f"keyframes/{chunk_id:04d}.jpg",
            "keyframe_ts": chunk.get("keyframe_ts", 0.0),
            "sampled_frames": chunk.get("sampled_frames", []),
        })

    narrative_data = {
        "video_id": video_id,
        "duration": duration,
        "num_chunks": len(new_chunks),
        "chunks": new_chunks,
    }
    with open(video_dir / "narrative.json", "w") as f:
        json.dump(narrative_data, f, ensure_ascii=False, indent=2)

    # Build metadata.json
    metadata = {
        "video_id": video_id,
        "duration": duration,
        "num_chunks": len(new_chunks),
        "fps": bank.get("fps", 1.0),
        "memory_kind": bank.get("memory_kind", "similarity_group"),
        "source": "copied_from_old_bank",
        "models": {
            "siglip_model": bank.get("siglip_model"),
            "vlm_caption_model": bank.get("vlm_caption_model"),
            "vlm_summary_model": bank.get("vlm_summary_model"),
            "bge_model": bank.get("bge_model"),
        },
    }
    with open(video_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Initialize empty attributes.json (will be filled by VLM later)
    attributes_data = {
        "video_id": video_id,
        "duration": duration,
        "num_chunks": len(new_chunks),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "static_attributes": [],   # 待生成
                "static_index_text": "",   # 待生成
            }
            for c in new_chunks
        ],
    }
    with open(video_dir / "attributes.json", "w") as f:
        json.dump(attributes_data, f, ensure_ascii=False, indent=2)

    # Remove old structure: episodic/ folder and main {video_id}.json
    shutil.rmtree(episodic_dir, ignore_errors=True)
    main_json.unlink()

    return True, f"{len(new_chunks)} chunks"


def main():
    video_ids = load_target_video_ids()
    print(f"目标视频数: {len(video_ids)}")
    print(f"目录: {BANK_DIR}")
    print()

    success, failed = 0, []
    for i, vid in enumerate(video_ids):
        ok, msg = reorganize_one(vid)
        if ok:
            success += 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"[{i+1}/{len(video_ids)}] {vid}: {msg}")
        else:
            failed.append((vid, msg))
            print(f"[{i+1}/{len(video_ids)}] {vid}: FAILED - {msg}")

    print(f"\n完成: {success}/{len(video_ids)}")
    if failed:
        print(f"失败 {len(failed)} 个: {failed[:5]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
