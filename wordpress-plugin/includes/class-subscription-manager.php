<?php

class Marketing_AI_Subscription_Manager {
    public function get_current_plan() {
        return get_option('marketing_ai_subscription_plan', 'free');
    }
    
    public function update_plan($plan_id) {
        update_option('marketing_ai_subscription_plan', sanitize_text_field($plan_id));
    }
    
    public function check_quota($feature) {
        // Fetch from backend
        $api = new Marketing_AI_API_Client();
        $res = $api->verify_api_key();
        
        if (is_wp_error($res) || !$res) return false;
        
        // Very basic quota check hook - the backend enforces mostly
        return true; 
    }
    
    public function subscribe_stripe($plan_id) {
        $api = new Marketing_AI_API_Client();
        return $api->subscribe('stripe', $plan_id, get_option('admin_email'));
    }

    public function subscribe_pesapal($plan_id) {
        $api = new Marketing_AI_API_Client();
        return $api->subscribe('pesapal', $plan_id, get_option('admin_email'));
    }
}
