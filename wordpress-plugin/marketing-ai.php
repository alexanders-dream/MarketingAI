<?php
/**
 * Plugin Name:       Marketing AI
 * Plugin URI:        https://marketingai.com/wordpress-plugin
 * Description:       AI-powered marketing content generation for WordPress. Create blog posts, social media content, and marketing copy with advanced AI.
 * Version:           1.0.0
 * Requires at least: 5.8
 * Requires PHP:      7.4
 * Author:            Your Name
 * Author URI:        https://yoursite.com
 * License:           GPL v2 or later
 * License URI:       https://www.gnu.org/licenses/gpl-2.0.html
 * Text Domain:       marketing-ai
 * Domain Path:       /languages
 */

// If this file is called directly, abort.
if (!defined('WPINC')) {
    die;
}

// Plugin version
define('MARKETING_AI_VERSION', '1.0.0');
define('MARKETING_AI_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('MARKETING_AI_PLUGIN_URL', plugin_dir_url(__FILE__));
define('MARKETING_AI_API_BASE_URL', 'https://api.marketingai.com/v1');

/**
 * The code that runs during plugin activation.
 */
function activate_marketing_ai() {
    require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-activator.php';
    Marketing_AI_Activator::activate();
}

/**
 * The code that runs during plugin deactivation.
 */
function deactivate_marketing_ai() {
    require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-deactivator.php';
    Marketing_AI_Deactivator::deactivate();
}

register_activation_hook(__FILE__, 'activate_marketing_ai');
register_deactivation_hook(__FILE__, 'deactivate_marketing_ai');

/**
 * The core plugin class
 */
require MARKETING_AI_PLUGIN_DIR . 'includes/class-marketing-ai.php';

/**
 * Begins execution of the plugin.
 */
function run_marketing_ai() {
    $plugin = new Marketing_AI();
    $plugin->run();
}
run_marketing_ai();
