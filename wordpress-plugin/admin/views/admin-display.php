<div class="wrap mkai-wrap">
    <h2>Marketing AI - Setup</h2>
    
    <div class="mkai-card">
        <h3>API Configuration</h3>
        <p>Enter your MarketingAI API key to connect this site to your subscription.</p>
        <form method="post" action="options.php">
            <?php
                settings_fields('marketing_ai_options');
                do_settings_sections('marketing_ai');
            ?>
            <table class="form-table">
                <tr valign="top">
                <th scope="row">API Key</th>
                <td>
                    <input type="password" name="marketing_ai_api_key" value="<?php echo esc_attr( get_option('marketing_ai_api_key') ); ?>" class="regular-text" />
                    <?php if (get_option('marketing_ai_api_key')): ?>
                        <span class="mkai-badge success">Connected</span>
                    <?php endif; ?>
                </td>
                </tr>
                <tr valign="top">
                <th scope="row">Webhook Secret</th>
                <td>
                    <input type="text" name="marketing_ai_webhook_secret" value="<?php echo esc_attr( get_option('marketing_ai_webhook_secret') ); ?>" class="regular-text" readonly />
                    <p class="description">Used automatically to secure communication from the MarketingAI backend.</p>
                </td>
                </tr>
                <tr valign="top">
                <th scope="row">Webhook URL</th>
                <td>
                    <input type="text" value="<?php echo esc_url(site_url('?mkai_hook=1')); ?>" class="regular-text" readonly />
                </td>
                </tr>
            </table>
            <?php submit_button('Save Configuration'); ?>
        </form>
    </div>
</div>

