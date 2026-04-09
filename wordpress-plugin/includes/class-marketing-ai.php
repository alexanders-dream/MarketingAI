<?php

class Marketing_AI {
    protected $plugin_name;
    protected $version;

    public function __construct() {
        $this->plugin_name = 'marketing-ai';
        $this->version = MARKETING_AI_VERSION;
    }

    public function run() {
        $this->define_admin_hooks();
        $this->define_public_hooks();
        $this->define_cron_hooks();
    }

    private function define_admin_hooks() {
        $admin_ui = new Marketing_AI_Admin_UI($this->plugin_name, $this->version);
        add_action('admin_menu', [$admin_ui, 'add_plugin_admin_menu']);
        add_action('admin_enqueue_scripts', [$admin_ui, 'enqueue_styles']);
        add_action('admin_enqueue_scripts', [$admin_ui, 'enqueue_scripts']);
        
        // AJAX hooks
        add_action('wp_ajax_marketing_ai_execute_task', [$admin_ui, 'ajax_execute_task']);
        add_action('wp_ajax_marketing_ai_generate_strategy', [$admin_ui, 'ajax_generate_strategy']);
    }

    private function define_public_hooks() {
        $webhook_receiver = new Marketing_AI_Webhook_Receiver();
        add_action('init', [$webhook_receiver, 'listen_for_webhooks']);
    }

    private function define_cron_hooks() {
        $cron_manager = new Marketing_AI_Cron_Manager();
        add_action('marketing_ai_execute_due_tasks', [$cron_manager, 'execute_due_tasks']);
        add_action('marketing_ai_auto_update_posts', [$cron_manager, 'auto_update_posts']);
    }
}
