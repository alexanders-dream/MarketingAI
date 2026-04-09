<?php
/**
 * Handles all API communication with MarketingAI backend.
 */
class Marketing_AI_API_Client {
    
    private $api_base_url;
    private $api_key;
    
    public function __construct() {
        $this->api_base_url = MARKETING_AI_API_BASE_URL;
        $this->api_key = get_option('marketing_ai_api_key', '');
    }
    
    /**
     * Make API request
     */
    public function request($endpoint, $method = 'GET', $body = null) {
        $url = $this->api_base_url . $endpoint;
        
        $args = [
            'method' => $method,
            'headers' => [
                'x-api-key' => $this->api_key,
                'Content-Type' => 'application/json',
            ],
            'timeout' => 60,
        ];
        
        if ($body !== null) {
            $args['body'] = json_encode($body);
        }
        
        $response = wp_remote_request($url, $args);
        
        if (is_wp_error($response)) {
            return $response;
        }
        
        $status_code = wp_remote_retrieve_response_code($response);
        $res_body = json_decode(wp_remote_retrieve_body($response), true);
        
        if ($status_code >= 400) {
            return new WP_Error(
                'api_error',
                $res_body['detail'] ?? 'API request failed',
                ['status' => $status_code]
            );
        }
        
        return $res_body;
    }
    
    public function verify_api_key() {
        $result = $this->request('/auth/verify', 'GET');
        return !is_wp_error($result);
    }
    
    public function get_usage_stats() {
        return $this->request('/auth/usage/stats', 'GET');
    }

    public function extract_context($data) {
        return $this->request('/context/extract', 'POST', $data);
    }
    
    public function get_context($context_id) {
        return $this->request("/context/{$context_id}", 'GET');
    }
    
    public function generate_strategy($context_id, $goal) {
        return $this->request('/strategy/generate', 'POST', ['context_id' => $context_id, 'goal' => $goal]);
    }
    
    public function execute_strategy($strategy_id) {
        return $this->request("/strategy/{$strategy_id}/execute", 'POST', []);
    }
    
    public function get_tasks($strategy_id = null) {
        $endpoint = '/tasks/';
        if ($strategy_id) {
            $endpoint .= "?strategy_id=" . urlencode($strategy_id);
        }
        return $this->request($endpoint, 'GET');
    }
    
    public function get_calendar() {
        return $this->request('/tasks/calendar', 'GET');
    }
    
    public function update_task($task_id, $status) {
        return $this->request("/tasks/{$task_id}", 'PATCH', ['status' => $status]);
    }
    
    public function execute_due_tasks() {
        return $this->request('/tasks/execute-due', 'POST'); 
    }
    
    public function generate_content($params) {
        return $this->request('/content/generate', 'POST', $params);
    }
    
    public function schedule_content($content_id, $date, $platform) {
        return $this->request('/content/schedule', 'POST', [
            'content_id' => $content_id, 'scheduled_date' => $date, 'platform' => $platform
        ]);
    }
    
    public function connect_social() {
        return $this->request('/social/accounts/connect', 'POST', []);
    }
    
    public function get_social_accounts() {
        return $this->request('/social/accounts', 'GET');
    }
    
    public function publish_social($content_id, $platforms) {
        return $this->request('/social/publish', 'POST', ['content_id' => $content_id, 'platforms' => $platforms]);
    }
    
    public function get_social_analytics($profile_id) {
        return $this->request("/social/analytics/{$profile_id}", 'GET');
    }
    
    public function subscribe($provider, $plan_id, $email, $phone = '') {
        return $this->request('/billing/subscribe', 'POST', [
            'payment_provider' => $provider,
            'plan_id' => $plan_id,
            'email' => $email,
            'phone' => $phone
        ]);
    }
}

