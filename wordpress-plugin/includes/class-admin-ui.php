<?php

class Marketing_AI_Admin_UI {
    private $plugin_name;
    private $version;

    public function __construct($plugin_name, $version) {
        $this->plugin_name = $plugin_name;
        $this->version = $version;
    }

    public function enqueue_styles($hook) {
        if (strpos($hook, 'marketing-ai') === false) return;
        wp_enqueue_style($this->plugin_name, MARKETING_AI_PLUGIN_URL . 'admin/css/admin.css', [], $this->version, 'all');
    }

    public function enqueue_scripts($hook) {
        if (strpos($hook, 'marketing-ai') === false) return;
        wp_enqueue_script($this->plugin_name, MARKETING_AI_PLUGIN_URL . 'admin/js/admin.js', ['jquery'], $this->version, true);
        wp_localize_script($this->plugin_name, 'marketingAiObj', ['ajax_url' => admin_url('admin-ajax.php'), 'nonce' => wp_create_nonce('marketing_ai_nonce')]);
    }

    public function add_plugin_admin_menu() {
        add_menu_page('Marketing AI', 'Marketing AI', 'manage_options', $this->plugin_name, [$this, 'display_dashboard'], 'dashicons-chart-pie', 25);
        add_submenu_page($this->plugin_name, 'Dashboard', 'Dashboard', 'manage_options', $this->plugin_name, [$this, 'display_dashboard']);
        add_submenu_page($this->plugin_name, 'Strategy', 'Strategy', 'manage_options', $this->plugin_name . '-strategy', [$this, 'display_strategy']);
        add_submenu_page($this->plugin_name, 'Task Calendar', 'Task Calendar', 'manage_options', $this->plugin_name . '-calendar', [$this, 'display_calendar']);
        add_submenu_page($this->plugin_name, 'Social Accounts', 'Social Accounts', 'manage_options', $this->plugin_name . '-social', [$this, 'display_social']);
        add_submenu_page($this->plugin_name, 'Analytics', 'Analytics', 'manage_options', $this->plugin_name . '-analytics', [$this, 'display_analytics']);
        add_submenu_page($this->plugin_name, 'Settings', 'Settings', 'manage_options', $this->plugin_name . '-settings', [$this, 'display_settings']);
    }

    public function display_settings() { require_once MARKETING_AI_PLUGIN_DIR . 'admin/views/admin-display.php'; }
    public function display_dashboard() { require_once MARKETING_AI_PLUGIN_DIR . 'admin/views/dashboard.php'; }
    public function display_strategy() { require_once MARKETING_AI_PLUGIN_DIR . 'admin/views/strategy.php'; }
    public function display_calendar() { require_once MARKETING_AI_PLUGIN_DIR . 'admin/views/task-calendar.php'; }
    public function display_social() { require_once MARKETING_AI_PLUGIN_DIR . 'admin/views/social-accounts.php'; }
    public function display_analytics() { require_once MARKETING_AI_PLUGIN_DIR . 'admin/views/analytics.php'; }

    public function ajax_execute_task() {
        check_ajax_referer('marketing_ai_nonce', 'nonce');
        $task_id = sanitize_text_field($_POST['task_id']);
        $api = new Marketing_AI_API_Client();
        $res = $api->update_task($task_id, 'approved');
        wp_send_json_success($res);
    }
    
    public function ajax_generate_strategy() {
        check_ajax_referer('marketing_ai_nonce', 'nonce');
        $goal = sanitize_text_field($_POST['goal']);
        // Here we would extract context, get ID, then generate strategy
        $extractor = new Marketing_AI_Context_Extractor();
        $ctx_data = $extractor->extract_site_context();
        
        $api = new Marketing_AI_API_Client();
        $ctx_res = $api->extract_context($ctx_data);
        if (is_wp_error($ctx_res)) {
            wp_send_json_error($ctx_res->get_error_message());
        }
        
        $strat_res = $api->generate_strategy($ctx_res['context_id'], $goal);
        if (is_wp_error($strat_res)) {
            wp_send_json_error($strat_res->get_error_message());
        }
        
        // Automatically execute it
        $exec_res = $api->execute_strategy($strat_res['id']);
        wp_send_json_success($exec_res);
    }
}
