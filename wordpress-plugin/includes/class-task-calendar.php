<?php

class Marketing_AI_Task_Calendar {
    public function get_upcoming_tasks() {
        $api = new Marketing_AI_API_Client();
        $tasks = $api->get_tasks();
        if (is_wp_error($tasks)) return [];
        return $tasks;
    }
    
    public function update_task_status($task_id, $status) {
        $api = new Marketing_AI_API_Client();
        return $api->update_task($task_id, $status);
    }
    
    public function get_calendar_view() {
        $api = new Marketing_AI_API_Client();
        return $api->get_calendar();
    }
}
