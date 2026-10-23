"""
Command-line interface for DesertFox OSINT & Privacy Engine
"""

import argparse
import sys
import os
import json
import numpy as np
from desertfox.pipeline.engine import DesertFoxEngine
from desertfox.privacy.takedown import TakedownNoticeGenerator

def main():
    parser = argparse.ArgumentParser(
        description="DesertFox: Open-Source Facial OSINT & Identity Exposure Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: index-demo
    idx_parser = subparsers.add_parser("index-demo", help="Generate and index synthetic sample records to verify pipeline")
    idx_parser.add_argument("--count", type=int, default=5, help="Number of sample identities to generate")
    idx_parser.add_argument("--store", type=str, default="./data/desertfox.index", help="Output path for vector store")

    # Subcommand: search-demo
    search_parser = subparsers.add_parser("search-demo", help="Run a test query against the indexed vector store")
    search_parser.add_argument("--store", type=str, default="./data/desertfox.index", help="Path to vector store index")
    search_parser.add_argument("--threshold", type=float, default=0.60, help="Similarity threshold (0.0 to 1.0)")

    # Subcommand: takedown
    td_parser = subparsers.add_parser("takedown", help="Generate a formal GDPR / DMCA erasure request")
    td_parser.add_argument("--url", type=str, required=True, help="URL where unauthorized image was located")
    td_parser.add_argument("--name", type=str, required=True, help="Subject's full legal name")
    td_parser.add_argument("--email", type=str, required=True, help="Subject's contact email")
    td_parser.add_argument("--type", choices=["gdpr", "dmca"], default="gdpr", help="Type of legal notice")

    args = parser.parse_args()

    if args.command == "index-demo":
        engine = DesertFoxEngine()
        print(f"[*] Generating {args.count} sample identity embeddings...")
        np.random.seed(42)
        for i in range(1, args.count + 1):
            # Generate deterministic sample image matrix (e.g. 112x112x3)
            sample_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
            rec_id = f"target_person_{i:03d}"
            url = f"https://leaked-profiles.example.org/users/{rec_id}.jpg"
            engine.index_image_array(sample_img, record_id=rec_id, source_url=url, metadata={"domain": "example.org", "tags": ["social_profile"]})
            print(f"    [+] Indexed {rec_id} -> {url}")

        engine.save_index(args.store)
        print(f"[✓] Successfully indexed {args.count} records. Saved to: {args.store}")

    elif args.command == "search-demo":
        if not os.path.exists(args.store):
            print(f"[!] Error: Vector store index '{args.store}' not found. Run 'index-demo' first.")
            sys.exit(1)

        engine = DesertFoxEngine()
        engine.load_index(args.store)
        print(f"[*] Loaded index with {len(engine.vector_store.records)} records from {args.store}")

        # Run query using target_person_001's deterministic seed
        np.random.seed(42)
        query_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        print(f"[*] Querying probe face against database (threshold={args.threshold})...")
        matches = engine.search_probe_array(query_img, top_k=5, threshold=args.threshold)

        if not matches:
            print("[-] No matching face exposure found above threshold.")
        else:
            print(f"[✓] Found {len(matches)} match(es):")
            for m in matches:
                print(f"    - Match: {m.record_id} | Similarity: {m.similarity * 100:.1f}%")
                print(f"      URL: {m.source_url}")
                print(f"      Metadata: {m.metadata}")

    elif args.command == "takedown":
        print("[*] Generating takedown notice...\n")
        if args.type == "gdpr":
            notice = TakedownNoticeGenerator.generate_gdpr_erasure_request(
                subject_name=args.name,
                subject_email=args.email,
                infringing_url=args.url
            )
        else:
            notice = TakedownNoticeGenerator.generate_dmca_notice(
                copyright_owner=args.name,
                contact_email=args.email,
                infringing_url=args.url
            )
        print("=" * 60)
        print(notice)
        print("=" * 60)

if __name__ == "__main__":
    main()
