<?php

class Marketing_AI_Activator {
    public static function activate() {
        // Require PHP 7.4+
        if (version_compare(PHP_VERSION, '7.4', '<')) {
            die('Marketing AI requires PHP version 7.4 or higher.');
        }

        // Schedule cron events
        if (!wp_next_scheduled('marketing_ai_execute_due_tasks')) {
            wp_schedule_event(time(), 'hourly', 'marketing_ai_execute_due_tasks');
        }
        if (!wp_next_scheduled('marketing_ai_auto_update_posts')) {
            wp_schedule_event(time(), 'daily', 'marketing_ai_auto_update_posts');
        }

        // Set default options
        if (get_option('marketing_ai_webhook_secret') === false) {
            update_option('marketing_ai_webhook_secret', wp_generate_password(32, false));
        }
    }
}
