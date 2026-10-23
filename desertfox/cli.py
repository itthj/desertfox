    elif args.command == "audit":
        from desertfox.config import EngineConfig
        engine = DesertFoxEngine(config=EngineConfig(vector_store_path=args.store))
        print(f"[*] Auditing host from {args.url} (max_pages={args.max_pages})")
        result = engine.audit_site(args.url, max_pages=args.max_pages, max_images=args.max_images)
        if result.get("status") != "success":
            print(f"[!] {result.get('message', 'Audit failed')}")
            sys.exit(1)
        print(f"[✓] Host {result['host']}: {result['pages_discovered']} pages (robots.txt ignored); indexed {result['newly_indexed_images']} images; all valid faces kept in scope")
        for page in result["pages"]:
            print(f"    - {page['url']} -> {page['indexed']} indexed")
