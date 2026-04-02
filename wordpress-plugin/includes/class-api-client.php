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
    private function request($endpoint, $method = 'GET', $body = null) {
        $url = $this->api_base_url . $endpoint;
        
        $args = [
            'method' => $method,
            'headers' => [
                'X-API-Key' => $this->api_key,
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
        $body = json_decode(wp_remote_retrieve_body($response), true);
        
        if ($status_code >= 400) {
            return new WP_Error(
                'api_error',
                $body['detail'] ?? 'API request failed',
                ['status' => $status_code]
            );
        }
        
        return $body;
    }
    
    /**
     * Create business context
     */
    public function create_context($context_data) {
        return $this->request('/contexts', 'POST', $context_data);
    }
    
    /**
     * Get business context
     */
    public function get_context($context_id) {
        return $this->request("/contexts/{$context_id}", 'GET');
    }
    
    /**
     * Update business context
     */
    public function update_context($context_id, $updates) {
        return $this->request("/contexts/{$context_id}", 'PATCH', $updates);
    }
    
    /**
     * Generate content
     */
    public function generate_content($params) {
        return $this->request('/content/generate', 'POST', $params);
    }
    
    /**
     * Check API key validity
     */
    public function verify_api_key() {
        $result = $this->request('/auth/verify', 'GET');
        return !is_wp_error($result);
    }
    
    /**
     * Get usage statistics
     */
    public function get_usage_stats() {
        return $this->request('/usage/stats', 'GET');
    }
}
