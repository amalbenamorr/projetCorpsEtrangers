"""
reset_memory.py — v2
────────────────────
Utility to reset or clean ChromaDB memory.

Usage:
  python reset_memory.py           # shows stats + asks confirmation
  python reset_memory.py --force   # resets without asking
  python reset_memory.py --stats   # shows stats only
  python reset_memory.py --clean   # removes unreliable cases (HUMAN_REVIEW + confidence=low)

Run when:
- You had bugs and wrote incorrect cases
- Memory is biasing decisions incorrectly
- After major tool fixes (like v5)
"""

import argparse
import json
from pathlib import Path

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

MEMORY_DIR      = "memory"
COLLECTION_NAME = "foreign_object_cases"


def get_stats():
    if not CHROMA_OK:
        print("❌ chromadb not installed")
        return None
    try:
        client     = chromadb.PersistentClient(path=MEMORY_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        count      = collection.count()
        if count == 0:
            print("📭 Memory is empty")
            return {"count": 0, "cases": []}

        results = collection.get(include=["metadatas"], limit=min(count, 100))
        metas   = results.get("metadatas", [])
        ids     = results.get("ids", [])  # ids sont retournés automatiquement

        print(f"\n📊 Memory stats: {count} total cases")
        print(f"{'─'*50}")

        decisions  = {}
        objects    = {}
        unreliable = []

        for m, cid in zip(metas, ids):
            d = m.get("decision", "unknown")
            o = m.get("object_detected", "unknown")
            c = m.get("confidence", "low")
            decisions[d] = decisions.get(d, 0) + 1
            objects[o]   = objects.get(o, 0) + 1
            if d == "HUMAN_REVIEW" and c == "low":
                unreliable.append(cid)

        print("Decisions:")
        for d, n in sorted(decisions.items()):
            print(f"  {d}: {n}")
        print("Objects detected:")
        for o, n in sorted(objects.items()):
            print(f"  {o}: {n}")
        if unreliable:
            print(f"\n⚠️  Unreliable cases (HUMAN_REVIEW + confidence=low): {len(unreliable)}")
        print(f"{'─'*50}")
        return {"count": count, "cases": metas, "ids": ids, "unreliable_ids": unreliable}
    except Exception as e:
        print(f"❌ Error reading memory: {e}")
        return None


def clean_unreliable():
    """Remove only HUMAN_REVIEW + confidence=low cases."""
    if not CHROMA_OK:
        print("❌ chromadb not installed")
        return False
    try:
        client     = chromadb.PersistentClient(path=MEMORY_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        count      = collection.count()
        if count == 0:
            print("Memory already empty.")
            return True

        results = collection.get(include=["metadatas", "ids"], limit=count)
        metas   = results.get("metadatas", [])
        ids     = results.get("ids", [])

        to_delete = []
        for m, cid in zip(metas, ids):
            if m.get("decision") == "HUMAN_REVIEW" and m.get("confidence", "low") == "low":
                to_delete.append(cid)

        if not to_delete:
            print("✅ No unreliable cases found. Memory is clean.")
            return True

        collection.delete(ids=to_delete)
        print(f"✅ Removed {len(to_delete)} unreliable cases.")
        print(f"   Remaining: {collection.count()} cases")
        return True
    except Exception as e:
        print(f"❌ Clean failed: {e}")
        return False


def reset_memory():
    if not CHROMA_OK:
        print("❌ chromadb not installed")
        return False
    try:
        client = chromadb.PersistentClient(path=MEMORY_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"✅ Collection '{COLLECTION_NAME}' deleted")
        except Exception:
            print(f"ℹ️  Collection did not exist")

        client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Empty collection '{COLLECTION_NAME}' created")

        for log in ["logs/memory_writes.log"]:
            p = Path(log)
            if p.exists():
                p.write_text("")
                print(f"✅ Cleared {log}")
        return True
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Reset ELMAZRAA ChromaDB memory")
    parser.add_argument("--force", action="store_true", help="Reset without confirmation")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--clean", action="store_true",
                        help="Remove only unreliable cases (HUMAN_REVIEW + confidence=low)")
    args = parser.parse_args()

    stats = get_stats()

    if args.stats:
        return

    if stats is None:
        return

    if args.clean:
        clean_unreliable()
        return

    if stats["count"] == 0:
        print("Nothing to reset.")
        return

    if not args.force:
        resp = input(f"\n⚠️  Reset ALL {stats['count']} cases? (yes/no): ").strip().lower()
        if resp != "yes":
            print("Cancelled.")
            return

    if reset_memory():
        print(f"\n🎯 Memory reset complete. Agent starts fresh.")
    else:
        print("\n❌ Reset failed.")


if __name__ == "__main__":
    main()