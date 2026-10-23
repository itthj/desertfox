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
        elif args.type == "ccpa":
            notice = TakedownNoticeGenerator.generate_ccpa_deletion_request(
                consumer_name=args.name,
                consumer_email=args.email,
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

    elif args.command == "audit":
        from desertfox.config import EngineConfig
        engine = DesertFoxEngine(config=EngineConfig(vector_store_path=args.store))
        print(f"[*] Auditing host from {args.url} (max_pages={args.max_pages})")
        result = engine.audit_site(args.url, max_pages=args.max_pages, max_images=args.max_images)
        if result.get("status") != "success":
            print(f"[!] {result.get('message', 'Audit failed')}")
            sys.exit(1)
        print(f"[✓] Host {result['host']}: {result['pages_discovered']} pages (robots.txt ignored); indexed {result['newly_indexed_images']} images; other faces discarded: {result.get('skipped_other_faces', 0)}")
        for page in result["pages"]:
            print(f"    - {page['url']} -> {page['indexed']} indexed")

    elif args.command == "enroll-reset":
        from desertfox.privacy.enrollment import FaceEnrollment
        FaceEnrollment(args.path).reset()
        print(f"[✓] Cleared enrolled face at {args.path}")

    elif args.command == "serve":
        import uvicorn
        print(f"[*] Starting DesertFox Cyber-OSINT HUD on http://{args.host}:{args.port}")
        uvicorn.run("desertfox.api.server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
