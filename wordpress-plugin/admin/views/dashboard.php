<div class="wrap mkai-wrap">
    <h2>Marketing AI - Dashboard</h2>
    <?php
        $api = new Marketing_AI_API_Client();
        $stats = $api->get_usage_stats();
        $is_connected = get_option('marketing_ai_api_key') ? true : false;
    ?>
    
    <?php if (!$is_connected): ?>
        <div class="notice notice-warning"><p>Please connect your API key in the Setup page.</p></div>
    <?php else: ?>
        <div class="mkai-dashboard-grid">
            <div class="mkai-card">
                <h3>Current Plan</h3>
                <?php $plan = get_option('marketing_ai_subscription_plan', 'Free'); ?>
                <span class="mkai-badge info"><?php echo esc_html(ucfirst($plan)); ?></span></p>
                <a href="?page=marketing-ai-settings" class="button">Upgrade Plan</a>
            </div>
            
            <div class="mkai-card">
                <h3>Usage This Month</h3>
                <ul>
                    <li>Content Generations: <?php echo esc_html($stats['content_generation'] ?? 0); ?></li>
                    <li>Social Posts: <?php echo esc_html($stats['social_posts'] ?? 0); ?></li>
                    <li>API Calls: <?php echo esc_html($stats['api_calls'] ?? 0); ?></li>
                </ul>
            </div>
        </div>
        
        <div class="mkai-card">
            <h3>Quick Actions</h3>
            <a href="?page=marketing-ai-strategy" class="button button-primary">Generate New Strategy</a>
            <a href="?page=marketing-ai-calendar" class="button">View Task Calendar</a>
        </div>
    <?php endif; ?>
</div>
