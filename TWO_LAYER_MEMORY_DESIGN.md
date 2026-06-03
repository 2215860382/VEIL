# Two-Layer Memory Bank Design

> Design discussion recorded 2026-06-03.

## Overview

Redesign the memory bank from a single-layer (similarity_group) to a two-layer architecture,
adding an online refresh mechanism for targeted evidence补全.

---

## 1. Architecture

Two layers per group (group_id = chunk_id):

| Layer | Name | Purpose |
|-------|------|---------|
| Layer 1 | Dynamic Narrative Memory | Events, actions, causal sequences, state changes |
| Layer 2 | Static Attribute Memory | Colors, numbers, OCR text, objects, appearance, spatial layout |

- Both layers share the same `group_id` (1:1 correspondence)
- Online refresh: temporary evidence for current question only, never written back to offline bank

---

## 2. Temporal Grouping

Keep existing: frame extraction + SigLIP embedding + cosine-similarity grouping.
Each group carries: `group_id`, `video_id`, `start_time`, `end_time`, `frame_ids`, `aligned_subtitles`.

---

## 3. Dynamic Narrative Memory

**Source**: keyframe image (VLM) + per-frame episodic captions + subtitles → structured narrative

**Key fields added to `MemoryChunk`**:

```python
key_events: List[str]          # ordered list of key events
actors: List[str]              # named persons or consistent descriptors
state_changes: List[str]       # "X changed from A to B" (empty if static)
temporal_relations: List[str]  # "first..., then..., finally..."
causal_clues: List[str]        # "because..., in order to..., which causes..."
```

**VLM Prompt (`DYNAMIC_SYS`)**: keyframe image + episodic_descs context.
- Lead with numbers/scores/counts in sentence 1
- Use delta notation "from X to Y" for changes
- Preserve all numbers, names, on-screen text exactly
- Extract key_events / actors / state_changes / temporal_relations / causal_clues as JSON

**Index text** = `summary + key_events + actors + actions + state_changes + temporal_relations + causal_clues + subtitles`
**Vector** = `v_semantic` (BGE-M3 on index text) — kept under existing field name

---

## 4. Static Attribute Memory (Visual Attribute Memory)

**Source**: keyframe image(s) per group → VLM attribute extraction

**New data class `StaticAttributeFrame`**:

```python
frame_id: str
timestamp: float
image_path: str
ocr_text: List[str]            # verbatim on-screen text
numbers: List[str]             # every visible number
colors: List[str]              # dominant colors
objects: List[str]             # specific object names
object_attributes: List[dict]  # {object, attributes: [color, shape, material, texture]}
people_appearance: List[str]   # physical descriptions
clothing: List[str]            # clothing items with colors/styles
spatial_layout: List[str]      # relative positions
textures: List[str]
scene_attributes: List[str]
```

**New fields in `MemoryChunk`**:

```python
static_frames: List[StaticAttributeFrame]
static_index_text: str         # flattened for BGE indexing
v_static: List[float]          # BGE-M3 on static_index_text
```

**VLM Prompt (`STATIC_ATTR_SYS`)**: extract ONLY static visible attributes, no actions/events.

**Index text** = `OCR + numbers + colors + objects + object_attributes + people_appearance + clothing + spatial_layout + textures + scene_attributes`
**Vector** = `v_static` (BGE-M3)

---

## 5. Retrieval: Joint Weighted Search

Single retrieval call per query returns full group evidence (both layers).

**Query type classification** (keyword-based):

| Query type | Keywords | v_semantic weight | v_static weight |
|------------|----------|-------------------|-----------------|
| `dynamic` | event, action, when, sequence, what happened, describe | 0.8 | 0.2 |
| `static` | color, number, text, OCR, how many, count, name, what color, read | 0.2 | 0.8 |
| `mixed` | (default) | 0.5 | 0.5 |

Fallback: if `v_static` is empty (legacy bank), use `v_semantic` only.

**Evidence format per group** (sent to Verifier):

```
[{t0}s-{t1}s] {summary}
Events: {key_events} | Actors: {actors} | Changes: {state_changes}
Visual: OCR={ocr_text} | Numbers={numbers} | Colors={colors} | Objects={objects}
Speech: {subtitles}
```

Verifier uses rubric scoring on merged evidence (no layer distinction).

---

## 6. Online Layer Refresh (Temporary Evidence)

Triggered when Verifier returns `INSUFFICIENT` after `max_iter`.

```python
def online_refresh(bank, failed_criteria, vlm, embedder,
                   video_path=None) -> List[str]:
    # 1. Parse failed_criteria to identify missing info type (dynamic vs static)
    # 2. Identify relevant time ranges from verifier output
    # 3. Run VLM on relevant keyframes with criterion-conditioned prompt
    # 4. Return temporary evidence strings (not written back to bank)
```

---

## 7. Legacy Bank Migration

- Dynamic layer: inherit existing `memory_text` (summary) → re-generate enhanced fields with VLM on keyframes
- Static layer: use existing `keyframes/*.jpg` (1 per chunk) for VLM attribute extraction
- Both layers regenerated via temporary `upgrade_bank.py` script (to be deleted after experiment)
- `memory_kind_v2 = "two_layer_v1"` marks upgraded banks

---

## 8. Field Naming Convention

Existing fields (unchanged):
- `v_semantic` — BGE vector for dynamic narrative text
- `v_visual` — SigLIP keyframe image vector (for dedup / visual assist)
- `memory_text` — dynamic narrative summary (upgraded content, same field name)
- `asr`, `ocr` — kept for legacy compatibility

New fields:
- `v_static` — BGE vector for static attribute text
- `static_frames` — list of per-frame static attribute data
- `static_index_text` — flattened static attribute text for indexing
- `key_events`, `actors`, `state_changes`, `temporal_relations`, `causal_clues`

---

## 9. Files to Modify

| File | Type | Change |
|------|------|--------|
| `src/memory/core/schema.py` | Permanent | Add new fields and `StaticAttributeFrame` class |
| `src/memory/similarity.py` | Permanent | Add `DYNAMIC_SYS` + `STATIC_ATTR_SYS` prompts; build both layers |
| `experiments/core/veil.py` | Permanent | Joint weighted retrieval; online refresh hook |
| `src/memory/online_refresh.py` | Permanent | Online refresh module |
| `src/memory/upgrade_bank.py` | **Temporary** | Legacy bank migration (delete after experiment) |
