jQuery(document).ready(function($) {
    $('#btn-generate-strategy').on('click', function(e) {
        e.preventDefault();
        
        var goal = $('#mkai-goal').val();
        if (!goal) {
            alert('Please enter a goal');
            return;
        }
        
        var $btn = $(this);
        var $spinner = $('#mkai-strategy-spinner');
        
        $btn.prop('disabled', true);
        $spinner.addClass('is-active');
        
        $.ajax({
            url: marketingAiObj.ajax_url,
            type: 'POST',
            data: {
                action: 'marketing_ai_generate_strategy',
                nonce: marketingAiObj.nonce,
                goal: goal
            },
            success: function(response) {
                $spinner.removeClass('is-active');
                $btn.prop('disabled', false);
                if (response.success) {
                    $('#mkai-strategy-result').show();
                } else {
                    alert('Error: ' + response.data);
                }
            },
            error: function(xhr, status, error) {
                $spinner.removeClass('is-active');
                $btn.prop('disabled', false);
                alert('Request failed. See console.');
                console.error(error);
            }
        });
    });

    $('.action-approve-task').on('click', function(e) {
        e.preventDefault();
        var $btn = $(this);
        var taskId = $btn.data('id');
        
        $btn.text('Approving...').prop('disabled', true);
        
        $.ajax({
            url: marketingAiObj.ajax_url,
            type: 'POST',
            data: {
                action: 'marketing_ai_execute_task',
                nonce: marketingAiObj.nonce,
                task_id: taskId
            },
            success: function(response) {
                if (response.success) {
                    $btn.replaceWith('<span style="color:green">Approved!</span>');
                } else {
                    alert('Failed to approve');
                    $btn.text('Approve').prop('disabled', false);
                }
            }
        });
    });
});
