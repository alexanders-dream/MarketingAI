<?php

class Marketing_AI_Webhook_Receiver {
    public function listen_for_webhooks() {
        if (isset($_GET['mkai_hook']) && $_GET['mkai_hook'] === '1') {
            $payload = file_get_contents('php://input');
            $signature = isset($_SERVER['HTTP_X_MARKETINGAI_SIGNATURE']) ? $_SERVER['HTTP_X_MARKETINGAI_SIGNATURE'] : '';
            
            if (!$this->verify_signature($payload, $signature)) {
                status_header(401);
                die('Invalid signature');
            }
            
            $data = json_decode($payload, true);
            $this->process_webhook($data);
            
            status_header(200);
            die('OK');
        }
    }

    private function verify_signature($payload, $signature) {
        $secret = get_option('marketing_ai_webhook_secret');
        if (!$secret) return false;
        
        $expected = hash_hmac('sha256', $payload, $secret);
        return hash_equals($expected, $signature);
    }
    
    private function process_webhook($data) {
        if (!isset($data['event'])) return;
        
        switch ($data['event']) {
            case 'content.generated':
                $this->create_draft_post($data['payload']);
                break;
            case 'task.completed':
                // Handled in backend, but we can do local sync if needed.
                break;
            case 'subscription.activated':
            case 'subscription.renewed':
                update_option('marketing_ai_subscription_status', 'active');
                break;
        }
    }

    private function create_draft_post($payload) {
        if (empty($payload['title']) || empty($payload['content'])) return;
        $post_data = [
            'post_title' => sanitize_text_field($payload['title']),
            'post_content' => wp_kses_post($payload['content']),
            'post_status' => 'draft',
            'post_author' => get_current_user_id() ?: 1,
        ];
        wp_insert_post($post_data);
    }
}
