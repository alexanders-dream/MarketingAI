# WordPress Plugin - Marketing AI

This directory contains the WordPress plugin for Marketing AI integration.

## Quick Start

### Installation for Development

1. **Copy to WordPress plugins directory:**
   ```bash
   cp -r wordpress-plugin /path/to/wordpress/wp-content/plugins/marketing-ai
   ```

2. **Activate the plugin:**
   - Go to WordPress Admin > Plugins
   - Find "Marketing AI"
   - Click "Activate"

3. **Configure API Key:**
   - Go to Marketing AI > Settings
   - Enter your API key from https://api.marketingai.com
   - Save settings

### File Structure

```
wordpress-plugin/
├── marketing-ai.php          # Main plugin file
├── readme.txt                # WordPress.org readme
├── includes/
│   ├── class-api-client.php  # API communication
│   └── (more classes...)
├── admin/                    # Admin interface
├── blocks/                   # Gutenberg blocks
└── assets/                   # Images and icons
```

## Features

✅ **Gutenberg Block** - Generate content in block editor  
✅ **Classic Editor** - TinyMCE integration  
✅ **Auto Context Detection** - Extracts site information automatically  
✅ **Bulk Generator** - Create multiple posts at once  
✅ **Usage Dashboard** - Track API usage and stats  

## Development

### Requirements

- WordPress 5.8+
- PHP 7.4+
- Marketing AI API key

### Testing Locally

```bash
# Using wp-env
cd wordpress-plugin
npx @wordpress/env start

# Or using Local by Flywheel, XAMPP, etc.
```

### Building Gutenberg Blocks

```bash
cd blocks/content-generator
npm install
npm run build
```

## Distribution

### WordPress.org (Free Version)

1. Prepare plugin following WordPress.org guidelines
2. Create SVN repository
3. Submit for review
4. Publish to directory

### Premium Version

- Sell on your own site
- List on CodeCanyon
- Offer via affiliate program

## Documentation

See `wordpress_plugin_guide.md` in the brain directory for complete implementation details.

## Support

- Documentation: https://docs.marketingai.com
- Support: support@marketingai.com
- Issues: https://github.com/yourrepo/issues

## License

GPL v2 or later
