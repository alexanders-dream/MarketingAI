
try:
    from firecrawl import FirecrawlApp
    print("Import successful")
    app = FirecrawlApp(api_key="test")
    print(f"Methods: {dir(app)}")
except Exception as e:
    print(f"Error: {e}")
