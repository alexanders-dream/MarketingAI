<?php

class Marketing_AI_Deactivator {
    public static function deactivate() {
        wp_clear_scheduled_hook('marketing_ai_execute_due_tasks');
        wp_clear_scheduled_hook('marketing_ai_auto_update_posts');
        // Do NOT delete options - user may reactivate
    }
}
