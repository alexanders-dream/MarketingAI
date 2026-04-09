<div class="wrap mkai-wrap">
    <h2>Marketing AI - Task Calendar</h2>
    
    <?php
        $calendar = new Marketing_AI_Task_Calendar();
        $tasks = $calendar->get_upcoming_tasks();
    ?>
    
    <div class="mkai-card">
        <table class="wp-list-table widefat fixed striped mkai-table">
            <thead>
                <tr>
                    <th>Date Scheduled</th>
                    <th>Task Title</th>
                    <th>Agent</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <?php if (empty($tasks)): ?>
                    <tr><td colspan="5">No upcoming tasks found. Generate a strategy first!</td></tr>
                <?php else: ?>
                    <?php foreach ($tasks as $task): ?>
                        <tr>
                            <td><?php echo esc_html(date('M j, Y', strtotime($task['scheduled_date']))); ?></td>
                            <td><?php echo esc_html($task['title']); ?></td>
                            <td><span class="mkai-badge agent-<?php echo esc_attr($task['assigned_agent']); ?>"><?php echo esc_html(ucfirst($task['assigned_agent'])); ?></span></td>
                            <td><span class="mkai-badge status-<?php echo esc_attr($task['status']); ?>"><?php echo esc_html(str_replace('_', ' ', $task['status'])); ?></span></td>
                            <td>
                                <?php if ($task['status'] === 'pending_review'): ?>
                                    <button class="button action-approve-task" data-id="<?php echo esc_attr($task['id']); ?>">Approve</button>
                                <?php endif; ?>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                <?php endif; ?>
            </tbody>
        </table>
    </div>
</div>
