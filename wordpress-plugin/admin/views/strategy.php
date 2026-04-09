<div class="wrap mkai-wrap">
    <h2>Marketing AI - Strategy Generator</h2>
    
    <div class="mkai-card">
        <p>Define your primary marketing goal, and our AI orchestrator will analyze your site context (posts, products, categories) and generate a multi-agent execution pipeline.</p>
        
        <div class="mkai-form-group">
            <label for="mkai-goal">Primary Marketing Goal (e.g., 'Increase sales for new summer apparel line by 20%')</label>
            <textarea id="mkai-goal" rows="4" style="width:100%"></textarea>
        </div>
        
        <button id="btn-generate-strategy" class="button button-primary is-large">Generate & Execute Strategy</button>
        <span class="spinner" id="mkai-strategy-spinner"></span>
    </div>
    
    <div id="mkai-strategy-result" style="display:none; margin-top:20px;" class="notice notice-success">
        <p><strong>Success!</strong> Strategy generated and tasks deployed. <a href="?page=marketing-ai-calendar">View Calendar</a></p>
    </div>
</div>
