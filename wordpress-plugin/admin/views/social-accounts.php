<div class="wrap mkai-wrap">
    <h2>Marketing AI - Social Accounts</h2>
    
    <div class="mkai-card">
        <h3>Connect Upload-Post Profile</h3>
        <p>Integrate your social media accounts via Upload-Post. Your API will orchestrate cross-platform publishing directly.</p>
        
        <?php
            $api = new Marketing_AI_API_Client();
            $link_res = $api->connect_social();
        ?>
        
        <?php if (is_wp_error($link_res)): ?>
            <div class="notice notice-error"><p>Failed to retrieve OAuth URL: <?php echo esc_html($link_res->get_error_message()); ?></p></div>
        <?php else: ?>
            <a href="<?php echo esc_url($link_res['oauth_url']); ?>" class="button button-primary" target="_blank">Connect Accounts</a>
        <?php endif; ?>
    </div>
</div>
